import os
from kedro.framework.hooks import hook_impl


class CredentialsHook:
    @hook_impl
    def after_context_created(self, context) -> None:
        creds = context.config_loader["credentials"]
        os.environ.setdefault("DEEPSEEK_API_KEY",          creds["deepseek"]["api_key"])
        os.environ.setdefault("GOOGLE_API_KEY",             creds["google"]["api_key"])
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN",   creds["huggingface"]["api_token"])
        os.environ.setdefault("HF_TOKEN",                    creds["huggingface"]["api_token"])
