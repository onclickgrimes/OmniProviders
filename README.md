# OmniProviders

Sidecar local OpenAI-compatible para os providers usados pelo Video Studio.

O sidecar concentra autenticação, descoberta de modelos, tradução de protocolos e geração temporária de mídia. O aplicativo consumidor continua responsável por prompts de domínio, seleção de fallback, execução de tools e armazenamento definitivo dos projetos.

## Desenvolvimento

```powershell
python -m pip install -r requirements.txt
$env:OMNIPROVIDERS_API_KEY = "local-secret"
python run_omni_providers.py
```

O endpoint padrão é `http://127.0.0.1:7814/v1`. Configure um cliente OpenAI com essa base URL e use `OMNIPROVIDERS_API_KEY` como API key.

O Gemini API consulta o catálogo de modelos da conta em tempo real. O endpoint de publisher models do Vertex AI não oferece listagem; por isso o adapter Vertex combina um catálogo verificado de modelos Google com uma chamada real de validação da conta. Os IDs TTS também são suplementados a partir da documentação oficial, pois não aparecem em `models.list`.

O runtime não inclui Playwright, Selenium ou navegador empacotado.

## Flow

O adapter Flow usa transporte HTTP e somente contas importadas explicitamente
pelo usuário. Registros técnicos, tokens avulsos e pools definidos por ambiente
não entram no pool de geração.

O OmniProviders publica o formulário de runtime em
`GET /providers/flow/configuration` e persiste seus valores separadamente em
`GET|PUT|DELETE /providers/flow/settings`. Provider Accounts continuam em
`/providers/flow/accounts`; o ID legado `flow-scraping` é reservado e migrado
automaticamente para settings.

O captcha pode usar o navegador externo instalado no sistema ou um solver HTTP:
YesCaptcha, CapMonster, EZCaptcha ou CapSolver. Modelos Flow não são publicados
em `/v1/models` enquanto não houver uma conta de usuário e uma configuração de
captcha válidas.
