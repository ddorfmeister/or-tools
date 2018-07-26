// Copyright 2010-2017 Google
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dynamic_library.h"

DynamicLibrary::DynamicLibrary(const char* library_name)
  : library_name_(std::string(library_name)) {
#if defined(_MSC_VER)
  library_handle_ = static_cast<void*>(LoadLibrary(library_name));
#elif defined(__GNUC__)
  library_handle_ = dlopen(library_name, RTLD_NOW);
#endif

  if (library_handle_ == nullptr) {
    throw std::runtime_error("Error: could not load library " +
                              library_name_);
  }
}

DynamicLibrary::DynamicLibrary(const std::string& library_name)
  : DynamicLibrary::DynamicLibrary(library_name.c_str()) {}

DynamicLibrary::~DynamicLibrary() {
  if (library_handle_ == nullptr) return;

#if defined(_MSC_VER)
  FreeLibrary(static_cast<HINSTANCE>(library_handle_));
#elif defined(__GNUC__)
  dlclose(library_handle_);
#endif
}