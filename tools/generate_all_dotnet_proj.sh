#!/usr/bin/env bash
# usage: ./tools/generate_all_dotnet_csproj.sh
set -e

# Gets OR_TOOLS_MAJOR and OR_TOOLS_MINOR
DIR="${BASH_SOURCE%/*}"
if [[ ! -d "${DIR}" ]]; then
  DIR="${PWD}";
fi
# shellcheck disable=SC1090
. "${DIR}/../Version.txt"

###############
##  Cleanup  ##
###############
echo "Remove previous .[cf]sproj .sln files..."
rm -f examples/*/*.csproj
rm -f examples/*/*.fsproj
rm -f examples/*/*.sln
rm -f ortools/*/samples/*.csproj
rm -f ortools/*/samples/*.fsproj
rm -f ortools/*/samples/*.sln
echo "Remove previous .[cf]sproj .sln files...DONE"

################
##  Examples  ##
################
for FILE in examples/*/*.[cf]s ; do
  # if no files found do nothing
  [[ -e "$FILE" ]] || continue
  ./tools/generate_dotnet_proj.sh "$FILE"
done
###############
##  Samples  ##
###############
for FILE in ortools/*/samples/*.[cf]s ; do
  # if no files found do nothing
  [[ -e "$FILE" ]] || continue
  ./tools/generate_dotnet_proj.sh "$FILE"
done

###########
##  SLN  ##
###########
if hash dotnet 2>/dev/null; then
  SLN=Google.OrTools.Examples.sln
  echo "Generate ${SLN}..."
  pushd examples/dotnet
  dotnet new sln -n ${SLN%.sln}
  for i in *.*proj; do
    dotnet sln ${SLN} add "$i"
  done
  echo "Generate ${SLN}...DONE"
  popd

  SLN=Google.OrTools.Contrib.sln
  echo "Generate ${SLN}..."
  pushd examples/contrib
  dotnet new sln -n ${SLN%.sln}
  for i in *.*proj; do
    dotnet sln ${SLN} add "$i"
  done
  echo "Generate ${SLN}...DONE"
  popd
fi
# vim: set tw=0 ts=2 sw=2 expandtab:
