#pragma once
// maximum we ever support (compile-time constant)
constexpr int MAX_PW_LEN = 7;

// run-time knob (defaults to MAX_PW_LEN, is copied to GPU before launch)
extern int g_pw_len;
