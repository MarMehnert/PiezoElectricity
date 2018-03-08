FILE(REMOVE_RECURSE
  "CMakeFiles/clean_output"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/clean_output.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
