project(MlLib LANGUAGES CXX)

add_library(ml-lib STATIC)

add_library(MlLib::MlLib ALIAS ml-lib)

target_compile_features(ml-lib PUBLIC cxx_std_23)

add_subdirectory(tensor)