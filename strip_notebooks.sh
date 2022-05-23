#!/bin/bash

# =====================================================================================================================
#   FUNCTIONS
# =====================================================================================================================

# --- show_help -----------------------------------------------------------------------------------
show_help() {

    cat << EOF
SYNTAX:

  strip_notebooks.sh help     --> displays this text
  strip_notebooks.sh <DIR>    --> strips all output cells from .ipynb files in <DIR> and subdirs using 'nbstripout'.

EOF

}


# =====================================================================================================================
#   MAIN
# =====================================================================================================================

# --- argument parsing ----------------------------------------------------------------------------

# --- shortcut if user just wants help ---
if [ "$1" = "help" ]; then
    show_help
    exit 0
fi

# --- extract dir ---
if [ $# -gt 0 ]; then
    dir="$1"
    shift
else
    show_help
    exit -1
fi

# --- actual work ---------------------------------------------------------------------------------
echo "Stripping output cells from all notebooks in $dir ..."
for file in $(find $dir -name '*.ipynb');
do
    echo "  --> Processing $file ..."
    nbstripout $file
done;
