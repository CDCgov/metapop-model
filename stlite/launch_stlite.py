import micropip

if "pyarrow" in micropip.list_mock_packages():
    micropip.remove_mock_package("pyarrow")
    await micropip.install("pyarrow")
await micropip.install("emfs:polars-1.24.0-cp39-abi3-emscripten_3_1_58_wasm32.whl")
await micropip.install("emfs:griddler-0.2.0-py3-none-any.whl")
await micropip.install("emfs:metapop-0.4.0-py2.py3-none-any.whl")

import metapop.app as app

app()
