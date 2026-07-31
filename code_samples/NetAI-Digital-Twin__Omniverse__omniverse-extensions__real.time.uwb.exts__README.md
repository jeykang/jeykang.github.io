# Warning!
extension 활성화 시 exts/uwb.rtls/uwb/rtls 폴더에 my_config.json 파일이 있어야 함

![UWB_RTLS](https://github.com/user-attachments/assets/968a971d-7383-442d-9145-1303df4ee81d)

# Extension Project Template

This project was automatically generated.

- `app` - It is a folder link to the location of your *Omniverse Kit* based app.
- `exts` - It is a folder where you can add new extensions. It was automatically added to extension search path. (Extension Manager -> Gear Icon -> Extension Search Path).

Open this folder using Visual Studio Code. It will suggest you to install few extensions that will make python experience better.

Look for "uwb.rtls" extension in extension manager and enable it. Try applying changes to any python files, it will hot-reload and you can observe results immediately.

Alternatively, you can launch your app from console with this folder added to search path and your extension enabled, e.g.:

```
> app\omni.code.bat --ext-folder exts --enable company.hello.world
```
