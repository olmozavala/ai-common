# ai-common
A set of classes and functions that are common for multiple AI projects

# Usage

The idea of this repo is to use it as a submodule.
To include it in your project you need to:

1. Add it as a submodule together with *hycom-utils*
```shell
git submodule add git@github.com:olmozavala/ai_common.git ai_common
```
2. Recursively download all the files
```shell
git submodule update --init --recursive
```
3. Include the proper paths in your python files
```python
import sys
sys.path.append("ai_common/")
```
4. Depending on the IDE you are using you also need to include the `ai_common`.
