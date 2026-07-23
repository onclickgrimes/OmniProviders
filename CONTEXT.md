# OmniProviders

OmniProviders centraliza acesso local a modelos e geração de mídia sem conhecer os fluxos do aplicativo consumidor.

## Language

**Provider**:
Uma plataforma remota que autentica contas e executa modelos, como Kiro, Antigravity ou Google GenAI.
_Avoid_: Serviço de IA, backend

**Provider Account**:
Uma identidade autenticada, cadastrada explicitamente pelo usuário, cuja assinatura, tier e permissões determinam os modelos disponíveis. Somente Provider Accounts podem entrar em pools de geração.
_Avoid_: Credencial, login

**Provider Configuration**:
Configuração técnica de transporte ou runtime de um Provider, como método de captcha, proxy ou navegador. Não representa uma identidade do usuário e nunca pode entrar em pools de geração.
_Avoid_: Conta padrão, conta técnica

Provider Configuration é persistida separadamente de Provider Account e usa a
interface `/providers/{provider}/settings`; a interface
`/providers/{provider}/configuration` descreve seu schema.

**Model**:
Um modelo executável confirmado para uma Provider Account, identificado externamente como `provider:model`.
_Avoid_: Opção estática, modelo presumido

**Capability**:
Uma modalidade ou operação confirmada para um Model, como imagem de entrada, vídeo de saída ou tool calling nativo.
_Avoid_: Feature genérica, compatibilidade presumida

**Generation Job**:
Uma execução assíncrona de mídia com identidade, progresso, estado terminal e artefatos resultantes.
_Avoid_: Fila de cena, render do projeto

**Artifact**:
Um arquivo temporário produzido ou recebido pelo sidecar, sem nome ou destino de domínio definidos pelo aplicativo consumidor.
_Avoid_: Asset do projeto, mídia da cena
