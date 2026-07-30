/**
 * @file addon.cc
 * @brief Node-API surface for cuda_engine.node.
 *
 * Error convention (CONTRACTS section 4), enforced in every function here:
 *   - Expected failures - no GPU, bad size, unconfigured scene - return
 *     { ok: false, reason: string }.
 *   - Only caller bugs (wrong argument type or count) throw Napi::TypeError.
 *   - Nothing else crosses the boundary. The addon is built with
 *     NAPI_DISABLE_CPP_EXCEPTIONS, so every Napi call is checked rather than
 *     wrapped in try/catch.
 *
 * The argument extraction below is deliberately repetitive. A clever generic
 * validator would be shorter and much harder to audit, and this is the layer
 * where a missed check turns a JS typo into a native crash.
 */

#include <napi.h>

#include <cstring>
#include <string>

#include "engine.h"
#include "native_view.h"

namespace {

using geoswarm::Engine;
using geoswarm::GetEngine;
using geoswarm::GetNativeView;
using geoswarm::InputUniforms;
using geoswarm::NativeView;
using geoswarm::SceneFromString;
using geoswarm::SceneId;
using geoswarm::SceneParams;
using geoswarm::Status;

/* ====================================================================== *
 *  Small helpers
 * ====================================================================== */

/** @brief Build { ok: true }. */
Napi::Object MakeOk(Napi::Env env) {
  Napi::Object o = Napi::Object::New(env);
  o.Set("ok", Napi::Boolean::New(env, true));
  return o;
}

/** @brief Build { ok: false, reason }. */
Napi::Object MakeFail(Napi::Env env, const std::string& reason) {
  Napi::Object o = Napi::Object::New(env);
  o.Set("ok", Napi::Boolean::New(env, false));
  o.Set("reason", Napi::String::New(env, reason));
  return o;
}

/**
 * @brief Read an optional numeric property, clamped and defaulted.
 *
 * Anything that is not a finite number (missing, null, a string, NaN) yields
 * the fallback rather than a garbage cast.
 */
double GetNumberOr(const Napi::Object& obj, const char* key, double fallback) {
  if (!obj.Has(key)) return fallback;
  Napi::Value v = obj.Get(key);
  if (!v.IsNumber()) return fallback;
  const double d = v.As<Napi::Number>().DoubleValue();
  if (!(d == d)) return fallback;  // NaN
  return d;
}

/** @brief Read an optional boolean property. */
bool GetBoolOr(const Napi::Object& obj, const char* key, bool fallback) {
  if (!obj.Has(key)) return fallback;
  Napi::Value v = obj.Get(key);
  if (!v.IsBoolean()) return fallback;
  return v.As<Napi::Boolean>().Value();
}

/**
 * @brief Read a uint32 count property with a hard ceiling.
 *
 * Negative and fractional values are rejected to zero, which the engine then
 * treats as "use the scene default".
 */
uint32_t GetCountOr(const Napi::Object& obj, const char* key, uint32_t fallback) {
  const double d = GetNumberOr(obj, key, static_cast<double>(fallback));
  if (d < 0.0) return 0;
  if (d > 4294967295.0) return 0;
  return static_cast<uint32_t>(d);
}

/**
 * @brief Copy a JS numeric array into a fixed-size float destination.
 *
 * Missing, wrong-typed and short arrays leave the destination untouched
 * (it is always pre-zeroed by the caller), so a malformed InputState degrades
 * to "no input" instead of reading past the end of something.
 *
 * @param src   the JS value expected to be an Array
 * @param dst   destination floats
 * @param n     number of components to copy
 */
void CopyFloatArray(const Napi::Value& src, float* dst, uint32_t n) {
  if (!dst || n == 0) return;
  if (!src.IsArray()) return;

  Napi::Array arr = src.As<Napi::Array>();
  const uint32_t len = arr.Length();
  const uint32_t take = (len < n) ? len : n;

  for (uint32_t i = 0; i < take; ++i) {
    Napi::Value e = arr.Get(i);
    if (!e.IsNumber()) continue;
    const double d = e.As<Napi::Number>().DoubleValue();
    if (!(d == d)) continue;  // skip NaN rather than poisoning the uniform
    dst[i] = static_cast<float>(d);
  }
}

/**
 * @brief Resolve an ArrayBuffer argument to a pointer + byte length.
 *
 * Accepts an ArrayBuffer or any TypedArray/DataView view over one, because JS
 * callers reach for whichever is convenient. Returns false when the argument
 * is neither - the caller then throws TypeError, which is correct: passing a
 * string where a buffer belongs is a caller bug.
 */
bool ResolveBuffer(const Napi::Value& v, void** outData, size_t* outBytes) {
  if (!outData || !outBytes) return false;
  *outData = nullptr;
  *outBytes = 0;

  if (v.IsArrayBuffer()) {
    Napi::ArrayBuffer ab = v.As<Napi::ArrayBuffer>();
    *outData = ab.Data();
    *outBytes = ab.ByteLength();
    return true;
  }
  if (v.IsTypedArray()) {
    Napi::TypedArray ta = v.As<Napi::TypedArray>();
    Napi::ArrayBuffer ab = ta.ArrayBuffer();
    // Honour the view's own offset and length - a Uint8Array over part of a
    // pooled buffer is a completely normal thing for the frame pump to send.
    *outData = static_cast<uint8_t*>(ab.Data()) + ta.ByteOffset();
    *outBytes = ta.ByteLength();
    return true;
  }
  if (v.IsDataView()) {
    Napi::DataView dv = v.As<Napi::DataView>();
    Napi::ArrayBuffer ab = dv.ArrayBuffer();
    *outData = static_cast<uint8_t*>(ab.Data()) + dv.ByteOffset();
    *outBytes = dv.ByteLength();
    return true;
  }
  return false;
}

/**
 * @brief Translate an InputState object from protocol.js into InputUniforms.
 *
 * Every field is optional. The destination starts fully zeroed and each field
 * is only written when it is present and the right type, so a partial or
 * hostile object can never produce an out-of-range uniform.
 */
void ParseInputState(const Napi::Object& obj, InputUniforms* out) {
  if (!out) return;
  std::memset(out, 0, sizeof(InputUniforms));

  // Defaults that matter if the caller omits the camera entirely - a zero
  // quaternion and zero fov would make the raster kernel degenerate.
  out->camPos[2] = 3.0f;
  out->camQuat[3] = 1.0f;
  out->fovYDeg = 50.0f;
  out->aspect = 16.0f / 9.0f;

  /* --- mouse ---------------------------------------------------------- */
  if (obj.Has("mouse")) {
    Napi::Value mv = obj.Get("mouse");
    if (mv.IsObject()) {
      Napi::Object m = mv.As<Napi::Object>();
      out->mouseX = static_cast<float>(GetNumberOr(m, "x", 0.0));
      out->mouseY = static_cast<float>(GetNumberOr(m, "y", 0.0));
      out->mouseDown = GetBoolOr(m, "down", false) ? 1 : 0;

      // mode: 1 attract, 2 repel, 3 vortex (protocol.js). Anything else is 0,
      // meaning "no pointer force".
      const int mode = static_cast<int>(GetNumberOr(m, "mode", 0.0));
      out->mouseMode = (mode >= 1 && mode <= 3) ? mode : 0;
    }
  }

  /* --- pointer raycast ------------------------------------------------ */
  if (obj.Has("pointerWorld")) {
    Napi::Value pv = obj.Get("pointerWorld");
    if (pv.IsArray()) {
      CopyFloatArray(pv, out->pointerWorld, 3);
      out->pointerValid = 1;
    }
    // null (a ray that missed the globe) leaves pointerValid at 0.
  }

  /* --- targets -------------------------------------------------------- */
  if (obj.Has("targets")) {
    Napi::Value tv = obj.Get("targets");
    if (tv.IsArray()) {
      Napi::Array arr = tv.As<Napi::Array>();
      const uint32_t len = arr.Length();
      // Silently drop anything past MAX_TARGETS rather than failing the frame;
      // the JS side is supposed to enforce the cap and this is the backstop.
      const uint32_t take = (len < GS_MAX_TARGETS) ? len : GS_MAX_TARGETS;

      uint32_t written = 0;
      for (uint32_t i = 0; i < take; ++i) {
        Napi::Value ev = arr.Get(i);
        if (!ev.IsObject()) continue;
        Napi::Object e = ev.As<Napi::Object>();

        CopyFloatArray(e.Get("pos"), out->targets[written].pos, 3);
        out->targets[written].strength = static_cast<float>(GetNumberOr(e, "strength", 1.0));
        out->targets[written].ttl = static_cast<float>(GetNumberOr(e, "ttl", 0.0));
        ++written;
      }
      out->targetCount = static_cast<int>(written);
    }
  }

  /* --- shockwaves ----------------------------------------------------- */
  if (obj.Has("shockwaves")) {
    Napi::Value sv = obj.Get("shockwaves");
    if (sv.IsArray()) {
      Napi::Array arr = sv.As<Napi::Array>();
      const uint32_t len = arr.Length();
      const uint32_t take = (len < GS_MAX_SHOCKWAVES) ? len : GS_MAX_SHOCKWAVES;

      uint32_t written = 0;
      for (uint32_t i = 0; i < take; ++i) {
        Napi::Value ev = arr.Get(i);
        if (!ev.IsObject()) continue;
        Napi::Object e = ev.As<Napi::Object>();

        CopyFloatArray(e.Get("pos"), out->shockwaves[written].pos, 3);
        out->shockwaves[written].age = static_cast<float>(GetNumberOr(e, "age", 0.0));
        ++written;
      }
      out->shockwaveCount = static_cast<int>(written);
    }
  }

  /* --- camera --------------------------------------------------------- */
  if (obj.Has("camera")) {
    Napi::Value cv = obj.Get("camera");
    if (cv.IsObject()) {
      Napi::Object c = cv.As<Napi::Object>();
      CopyFloatArray(c.Get("pos"), out->camPos, 3);
      CopyFloatArray(c.Get("quat"), out->camQuat, 4);  // xyzw, three.js order

      const double fov = GetNumberOr(c, "fovYDeg", 50.0);
      out->fovYDeg = (fov > 1.0 && fov < 179.0) ? static_cast<float>(fov) : 50.0f;

      const double aspect = GetNumberOr(c, "aspect", 16.0 / 9.0);
      out->aspect = (aspect > 0.01 && aspect < 100.0) ? static_cast<float>(aspect)
                                                      : (16.0f / 9.0f);

      // A zero-length quaternion rotates every ray to the origin. Renormalize
      // defensively; JS sends three.js quaternions which are unit, but a
      // hand-built InputState might not be.
      const float qx = out->camQuat[0], qy = out->camQuat[1];
      const float qz = out->camQuat[2], qw = out->camQuat[3];
      const float len2 = qx * qx + qy * qy + qz * qz + qw * qw;
      if (len2 < 1e-8f) {
        out->camQuat[0] = 0.0f;
        out->camQuat[1] = 0.0f;
        out->camQuat[2] = 0.0f;
        out->camQuat[3] = 1.0f;
      }
    }
  }

  /* --- appearance ------------------------------------------------------ */
  // Optional extra property on the input object (see InputUniforms::pointScale).
  // The renderer premultiplies the storm size slider with its pixel ratio; a
  // missing or out-of-range value falls back to 1 so an older renderer build
  // keeps rendering at the default size instead of at zero.
  {
    const double ps = GetNumberOr(obj, "pointScale", 1.0);
    out->pointScale = (ps > 0.05 && ps < 64.0) ? static_cast<float>(ps) : 1.0f;
  }

  /* --- clock ---------------------------------------------------------- */
  out->timeSec = static_cast<float>(GetNumberOr(obj, "timeSec", 0.0));
}

/* ====================================================================== *
 *  Exported: device + lifecycle
 * ====================================================================== */

/** @brief getDeviceInfo() -> { ok, name?, ccMajor?, ccMinor?, vramMB?, driverVersion?, reason? } */
Napi::Value GetDeviceInfo(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // No arguments to validate - this is explicitly safe to call on a machine
  // with no NVIDIA hardware at all.
  geoswarm::DeviceInfo d = Engine::QueryDevice();
  if (!d.ok) return MakeFail(env, d.reason);

  Napi::Object o = MakeOk(env);
  o.Set("name", Napi::String::New(env, d.name));
  o.Set("ccMajor", Napi::Number::New(env, d.ccMajor));
  o.Set("ccMinor", Napi::Number::New(env, d.ccMinor));
  o.Set("vramMB", Napi::Number::New(env, d.vramMB));
  o.Set("driverVersion", Napi::Number::New(env, d.driverVersion));
  return o;
}

/** @brief init() -> { ok, reason? } */
Napi::Value Init(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Status s = GetEngine().Init();
  return s.ok ? MakeOk(env) : MakeFail(env, s.reason);
}

/** @brief shutdown() -> undefined */
Napi::Value ShutdownFn(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  // Order matters: the view's render thread touches engine state, so it has to
  // be joined before the engine frees anything.
  GetNativeView().Destroy();
  GetEngine().Shutdown();
  return env.Undefined();
}

/* ====================================================================== *
 *  Exported: configuration
 * ====================================================================== */

/** @brief configureScene(scene, params) -> { ok, vramUsedMB?, reason? } */
Napi::Value ConfigureScene(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 2) {
    Napi::TypeError::New(env, "configureScene(scene: string, params: object) requires 2 arguments.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[0].IsString()) {
    Napi::TypeError::New(env, "configureScene: 'scene' must be a string.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[1].IsObject()) {
    Napi::TypeError::New(env, "configureScene: 'params' must be an object.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const std::string sceneName = info[0].As<Napi::String>().Utf8Value();
  const SceneId scene = SceneFromString(sceneName);
  if (scene == SceneId::kNone) {
    // An unknown scene name is a value problem, not a type problem - the
    // caller passed a string as documented, it just was not one we know.
    return MakeFail(env, "Unknown scene '" + sceneName + "' (expected swarm|weather|storm).");
  }

  Napi::Object params = info[1].As<Napi::Object>();
  SceneParams sp;
  sp.swarmCount = GetCountOr(params, "swarmCount", 0);
  sp.weatherGrid = GetCountOr(params, "weatherGrid", 0);
  sp.stormCount = GetCountOr(params, "stormCount", 0);

  geoswarm::ConfigureResult r = GetEngine().ConfigureScene(scene, sp);
  if (!r.ok) return MakeFail(env, r.reason);

  Napi::Object o = MakeOk(env);
  o.Set("vramUsedMB", Napi::Number::New(env, r.vramUsedMB));
  return o;
}

/** @brief setInput(input) -> undefined */
Napi::Value SetInput(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1) {
    Napi::TypeError::New(env, "setInput(input: object) requires 1 argument.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[0].IsObject()) {
    Napi::TypeError::New(env, "setInput: 'input' must be an object.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  InputUniforms uniforms;
  ParseInputState(info[0].As<Napi::Object>(), &uniforms);

  // Returns undefined per the contract. A failure here is logged native-side
  // rather than surfaced - this is called every frame and a per-frame object
  // allocation for a result nobody reads would be pure waste.
  Status s = GetEngine().SetInput(uniforms);
  if (!s.ok) {
    fprintf(stderr, "[cuda_engine] setInput failed: %s\n", s.reason.c_str());
  }
  return env.Undefined();
}

/** @brief uploadEarthTexture(rgba, w, h) -> { ok, reason? } */
Napi::Value UploadEarthTexture(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3) {
    Napi::TypeError::New(env, "uploadEarthTexture(rgba: ArrayBuffer, w: number, h: number) requires 3 arguments.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  void* data = nullptr;
  size_t bytes = 0;
  if (!ResolveBuffer(info[0], &data, &bytes)) {
    Napi::TypeError::New(env, "uploadEarthTexture: 'rgba' must be an ArrayBuffer or a typed array view.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[1].IsNumber() || !info[2].IsNumber()) {
    Napi::TypeError::New(env, "uploadEarthTexture: 'w' and 'h' must be numbers.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const int w = info[1].As<Napi::Number>().Int32Value();
  const int h = info[2].As<Napi::Number>().Int32Value();

  if (!data) return MakeFail(env, "Earth texture buffer is detached.");

  Status s = GetEngine().UploadEarthTexture(static_cast<const uint8_t*>(data), bytes, w, h);
  return s.ok ? MakeOk(env) : MakeFail(env, s.reason);
}

/* ====================================================================== *
 *  Exported: per-frame work
 * ====================================================================== */

/** @brief step(scene, dtMs, out) -> { ok, count, simMs, copyMs, reason? } */
Napi::Value Step(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 3) {
    Napi::TypeError::New(env, "step(scene: string, dtMs: number, out: ArrayBuffer) requires 3 arguments.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[0].IsString()) {
    Napi::TypeError::New(env, "step: 'scene' must be a string.").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[1].IsNumber()) {
    Napi::TypeError::New(env, "step: 'dtMs' must be a number.").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  void* out = nullptr;
  size_t outBytes = 0;
  if (!ResolveBuffer(info[2], &out, &outBytes)) {
    Napi::TypeError::New(env, "step: 'out' must be an ArrayBuffer or a typed array view.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const std::string sceneName = info[0].As<Napi::String>().Utf8Value();
  const SceneId scene = SceneFromString(sceneName);
  if (scene == SceneId::kNone) {
    return MakeFail(env, "Unknown scene '" + sceneName + "' (expected swarm|weather|storm).");
  }

  // A transferred (detached) ArrayBuffer reports zero length and a null data
  // pointer. That is a live-fire hazard with the transfer-based frame pump, so
  // name it explicitly instead of failing on "buffer too small".
  if (!out || outBytes == 0) {
    return MakeFail(env, "Output buffer is detached or empty - it was likely transferred away.");
  }

  const double dtMs = info[1].As<Napi::Number>().DoubleValue();

  geoswarm::StepResult r = GetEngine().Step(scene, dtMs, out, outBytes);
  if (!r.ok) return MakeFail(env, r.reason);

  Napi::Object o = MakeOk(env);
  o.Set("count", Napi::Number::New(env, static_cast<double>(r.count)));
  o.Set("simMs", Napi::Number::New(env, r.simMs));
  o.Set("copyMs", Napi::Number::New(env, r.copyMs));
  return o;
}

/** @brief getWeatherField(out) -> { ok, w, h, reason? } */
Napi::Value GetWeatherField(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1) {
    Napi::TypeError::New(env, "getWeatherField(out: ArrayBuffer) requires 1 argument.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  void* out = nullptr;
  size_t outBytes = 0;
  if (!ResolveBuffer(info[0], &out, &outBytes)) {
    Napi::TypeError::New(env, "getWeatherField: 'out' must be an ArrayBuffer or a typed array view.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!out || outBytes == 0) {
    return MakeFail(env, "Output buffer is detached or empty - it was likely transferred away.");
  }

  geoswarm::FieldResult r = GetEngine().GetWeatherField(out, outBytes);
  if (!r.ok) return MakeFail(env, r.reason);

  Napi::Object o = MakeOk(env);
  o.Set("w", Napi::Number::New(env, r.w));
  o.Set("h", Napi::Number::New(env, r.h));
  return o;
}

/**
 * @brief getGpuStats() -> { ok, vramUsedMB?, vramTotalMB?, gpuUtilPct?, memUtilPct?, reason? }
 *
 * No arguments, and safe to call at any point in the lifecycle - the overlay
 * polls it at ~1 Hz from before init() until after shutdown().
 *
 * Fields are set only when their source produced them, which is the whole point
 * of the shape: a machine whose driver has no NVML still gets vramUsedMB and
 * vramTotalMB with ok:true, and the absence of gpuUtilPct is how the JS side
 * knows to render that half of the line as unavailable rather than as 0%.
 */
Napi::Value GetGpuStats(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  geoswarm::GpuStats s = GetEngine().QueryGpuStats();
  if (!s.ok) return MakeFail(env, s.reason);

  Napi::Object o = MakeOk(env);
  if (s.hasVram) {
    o.Set("vramUsedMB", Napi::Number::New(env, s.vramUsedMB));
    o.Set("vramTotalMB", Napi::Number::New(env, s.vramTotalMB));
  }
  if (s.hasUtil) {
    o.Set("gpuUtilPct", Napi::Number::New(env, static_cast<double>(s.gpuUtilPct)));
    o.Set("memUtilPct", Napi::Number::New(env, static_cast<double>(s.memUtilPct)));
  }
  // A reason on an ok result is not a failure - it explains which optional
  // fields are missing and why (typically "no nvml.dll on this machine").
  if (!s.reason.empty()) {
    o.Set("reason", Napi::String::New(env, s.reason));
  }
  return o;
}

/** @brief renderFrame(scene, w, h, dtMs, out) -> { ok, simMs, renderMs, copyMs, reason? } */
Napi::Value RenderFrame(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5) {
    Napi::TypeError::New(env,
        "renderFrame(scene: string, w: number, h: number, dtMs: number, out: ArrayBuffer) requires 5 arguments.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[0].IsString()) {
    Napi::TypeError::New(env, "renderFrame: 'scene' must be a string.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[1].IsNumber() || !info[2].IsNumber() || !info[3].IsNumber()) {
    Napi::TypeError::New(env, "renderFrame: 'w', 'h' and 'dtMs' must be numbers.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  void* out = nullptr;
  size_t outBytes = 0;
  if (!ResolveBuffer(info[4], &out, &outBytes)) {
    Napi::TypeError::New(env, "renderFrame: 'out' must be an ArrayBuffer or a typed array view.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!out || outBytes == 0) {
    return MakeFail(env, "Output buffer is detached or empty - it was likely transferred away.");
  }

  const std::string sceneName = info[0].As<Napi::String>().Utf8Value();
  const SceneId scene = SceneFromString(sceneName);
  if (scene == SceneId::kNone) {
    return MakeFail(env, "Unknown scene '" + sceneName + "' (expected swarm|weather|storm).");
  }

  const int w = info[1].As<Napi::Number>().Int32Value();
  const int h = info[2].As<Napi::Number>().Int32Value();
  const double dtMs = info[3].As<Napi::Number>().DoubleValue();

  geoswarm::RenderResult r = GetEngine().RenderFrame(scene, w, h, dtMs, out, outBytes);
  if (!r.ok) return MakeFail(env, r.reason);

  Napi::Object o = MakeOk(env);
  o.Set("simMs", Napi::Number::New(env, r.simMs));
  o.Set("renderMs", Napi::Number::New(env, r.renderMs));
  o.Set("copyMs", Napi::Number::New(env, r.copyMs));
  return o;
}

/* ====================================================================== *
 *  Exported: native view
 * ====================================================================== */

/** @brief nativeViewCreate(hwnd: Buffer, x, y, w, h, dpr) -> { ok, reason? } */
Napi::Value NativeViewCreate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 6) {
    Napi::TypeError::New(env,
        "nativeViewCreate(hwnd: Buffer, x, y, w, h, dpr) requires 6 arguments.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[0].IsBuffer()) {
    Napi::TypeError::New(env,
        "nativeViewCreate: 'hwnd' must be the Buffer from BrowserWindow.getNativeWindowHandle().")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  for (int i = 1; i <= 5; ++i) {
    if (!info[i].IsNumber()) {
      Napi::TypeError::New(env, "nativeViewCreate: 'x', 'y', 'w', 'h' and 'dpr' must be numbers.")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
  }

  Napi::Buffer<uint8_t> handleBuf = info[0].As<Napi::Buffer<uint8_t>>();
  // Electron hands back a pointer-sized buffer; anything shorter would mean
  // reading past the end when we dereference it as an HWND.
  if (handleBuf.Length() < sizeof(HWND) || handleBuf.Data() == nullptr) {
    return MakeFail(env, "Window handle buffer is too small or detached.");
  }

  HWND parent = nullptr;
  std::memcpy(&parent, handleBuf.Data(), sizeof(HWND));
  if (!parent) return MakeFail(env, "Window handle buffer contains a null HWND.");

  const int x = info[1].As<Napi::Number>().Int32Value();
  const int y = info[2].As<Napi::Number>().Int32Value();
  const int w = info[3].As<Napi::Number>().Int32Value();
  const int h = info[4].As<Napi::Number>().Int32Value();
  const double dpr = info[5].As<Napi::Number>().DoubleValue();

  Status s = GetNativeView().Create(parent, x, y, w, h, dpr);
  return s.ok ? MakeOk(env) : MakeFail(env, s.reason);
}

/** @brief nativeViewSetRect(x, y, w, h, dpr) -> { ok } */
Napi::Value NativeViewSetRect(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5) {
    Napi::TypeError::New(env, "nativeViewSetRect(x, y, w, h, dpr) requires 5 arguments.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  for (int i = 0; i < 5; ++i) {
    if (!info[i].IsNumber()) {
      Napi::TypeError::New(env, "nativeViewSetRect: all arguments must be numbers.")
          .ThrowAsJavaScriptException();
      return env.Undefined();
    }
  }

  const int x = info[0].As<Napi::Number>().Int32Value();
  const int y = info[1].As<Napi::Number>().Int32Value();
  const int w = info[2].As<Napi::Number>().Int32Value();
  const int h = info[3].As<Napi::Number>().Int32Value();
  const double dpr = info[4].As<Napi::Number>().DoubleValue();

  Status s = GetNativeView().SetRect(x, y, w, h, dpr);
  return s.ok ? MakeOk(env) : MakeFail(env, s.reason);
}

/** @brief nativeViewSetVisible(visible: bool) -> undefined */
Napi::Value NativeViewSetVisible(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsBoolean()) {
    Napi::TypeError::New(env, "nativeViewSetVisible(visible: boolean) requires a boolean argument.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  GetNativeView().SetVisible(info[0].As<Napi::Boolean>().Value());
  return env.Undefined();
}

/** @brief nativeViewStart(scene, { vsync }) -> { ok, reason? } */
Napi::Value NativeViewStart(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1) {
    Napi::TypeError::New(env, "nativeViewStart(scene: string, opts?: object) requires at least 1 argument.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }
  if (!info[0].IsString()) {
    Napi::TypeError::New(env, "nativeViewStart: 'scene' must be a string.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  const std::string sceneName = info[0].As<Napi::String>().Utf8Value();
  const SceneId scene = SceneFromString(sceneName);
  if (scene == SceneId::kNone) {
    return MakeFail(env, "Unknown scene '" + sceneName + "' (expected swarm|weather|storm).");
  }

  // Options are optional; a missing object means vsync on, which is the safe
  // default (an unlocked loop on a stopped view would spin a core).
  bool vsync = true;
  if (info.Length() >= 2 && info[1].IsObject()) {
    vsync = GetBoolOr(info[1].As<Napi::Object>(), "vsync", true);
  }

  Status s = GetNativeView().Start(scene, vsync);
  return s.ok ? MakeOk(env) : MakeFail(env, s.reason);
}

/** @brief nativeViewStop() -> undefined */
Napi::Value NativeViewStop(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  GetNativeView().Stop();
  return env.Undefined();
}

/** @brief nativeViewStats() -> { fps, frameMs, simMs } */
Napi::Value NativeViewStats(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  geoswarm::ViewStats s = GetNativeView().Stats();
  Napi::Object o = Napi::Object::New(env);
  o.Set("fps", Napi::Number::New(env, s.fps));
  o.Set("frameMs", Napi::Number::New(env, s.frameMs));
  o.Set("simMs", Napi::Number::New(env, s.simMs));
  return o;
}

/* ====================================================================== *
 *  Module registration
 * ====================================================================== */

Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
  exports.Set("getDeviceInfo", Napi::Function::New(env, GetDeviceInfo));
  exports.Set("getGpuStats", Napi::Function::New(env, GetGpuStats));
  exports.Set("init", Napi::Function::New(env, Init));
  exports.Set("shutdown", Napi::Function::New(env, ShutdownFn));

  exports.Set("configureScene", Napi::Function::New(env, ConfigureScene));
  exports.Set("setInput", Napi::Function::New(env, SetInput));
  exports.Set("uploadEarthTexture", Napi::Function::New(env, UploadEarthTexture));

  exports.Set("step", Napi::Function::New(env, Step));
  exports.Set("getWeatherField", Napi::Function::New(env, GetWeatherField));
  exports.Set("renderFrame", Napi::Function::New(env, RenderFrame));

  exports.Set("nativeViewCreate", Napi::Function::New(env, NativeViewCreate));
  exports.Set("nativeViewSetRect", Napi::Function::New(env, NativeViewSetRect));
  exports.Set("nativeViewSetVisible", Napi::Function::New(env, NativeViewSetVisible));
  exports.Set("nativeViewStart", Napi::Function::New(env, NativeViewStart));
  exports.Set("nativeViewStop", Napi::Function::New(env, NativeViewStop));
  exports.Set("nativeViewStats", Napi::Function::New(env, NativeViewStats));

  return exports;
}

}  // namespace

NODE_API_MODULE(cuda_engine, InitModule)
