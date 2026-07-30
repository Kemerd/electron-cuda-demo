/**
 * @file scene_state.cc
 * @brief Definition of the process-wide SceneState registry.
 *
 * Deliberately a plain C++ TU with no CUDA in it: the registry is host-side
 * bookkeeping, and keeping it out of the .cu files means the pure-C++ side of
 * the addon can include scene_state.h without dragging in the CUDA headers.
 */

#include "scene_state.h"

namespace geoswarm {

SceneState& GetSceneState() {
  // Function-local static: zero-initialised before any launcher can run, and
  // never destroyed early (the atomics are trivially destructible anyway).
  static SceneState state;
  return state;
}

}  // namespace geoswarm
