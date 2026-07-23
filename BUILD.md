# Build do OmniProviders

O sidecar usa um ambiente virtual próprio para impedir que dependências do Core
vazem para o executável. O empacotamento é `onedir` com PyInstaller; assim o
cliente tipado do Google não precisa ser transpilado integralmente para C.

```powershell
.\scripts\setup.ps1
.\scripts\build_sidecar.ps1
```

O artefato esperado pelo Electron é:

`build/sidecar/run_omni_providers.dist/run_omni_providers.exe`

O build standalone inclui FastAPI, os transports HTTP/OAuth e os clientes
Google/Gemini Web. Não inclui Playwright, Selenium ou um navegador empacotado.
