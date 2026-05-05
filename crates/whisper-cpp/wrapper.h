// bindgen entry point. Pulls only what we use. Adding new
// whisper.cpp surface to the safe wrapper means adding the
// matching `#include` here AND extending the `allowlist_*`
// directives in `build.rs` — there is no implicit re-export.
#include "whisper.h"
