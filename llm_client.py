"""
llm_client.py — ClaudeTrader
─────────────────────────────────────────────────────────────────────
Cliente LLM que conecta con la API de Anthropic (Claude) para
generar decisiones de trading cuantitativo: BUY | SELL | HOLD.

La configuración del modelo se lee de  config/config.yaml  (sección ai_engine)
y la API key se obtiene de la variable de entorno  ANTHROPIC_API_KEY.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

# ── Cargar .env (idempotente si ya se llamó en main.py) ─────────────
load_dotenv()

logger = logging.getLogger("ClaudeTrader")

# ── System Prompt estricto ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a quantitative trading engine. "
    "You will receive market data including price action, volume, "
    "and technical indicators for a financial asset.\n\n"
    "RULES:\n"
    "1. Analyze the data objectively using quantitative reasoning.\n"
    "2. Your response MUST be exactly ONE of these three words: "
    "BUY, SELL, or HOLD.\n"
    "3. Do NOT include any explanation, punctuation, or additional text.\n"
    "4. Do NOT wrap the answer in quotes or code blocks.\n"
    "5. Respond with a single word only."
)


class ClaudeTrader:
    """
    Interfaz con la API de Anthropic para decisiones de trading.

    Uso básico::

        trader = ClaudeTrader()
        decision = trader.analyze_market_data({"close": 67200, ...})
        print(decision)  # → "BUY" | "SELL" | "HOLD"
    """

    # ----------------------------------------------------------------
    # Inicialización
    # ----------------------------------------------------------------
    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Carga la configuración del modelo desde config.yaml y la API key
        desde las variables de entorno.

        Parameters
        ----------
        config_path : str, optional
            Ruta al archivo config.yaml.  Si no se proporciona, se busca
            en  ``config/config.yaml``  relativo a la raíz del proyecto.
        """
        # 1. Resolver ruta del config
        if config_path is None:
            config_path = str(
                Path(__file__).resolve().parent / "config" / "config.yaml"
            )

        # 2. Leer configuración YAML
        ai_cfg = self._load_ai_config(config_path)
        self.model: str = ai_cfg.get("model", "claude-3-5-sonnet-20240620")
        self.max_tokens: int = ai_cfg.get("max_tokens", 1024)
        self.temperature: float = ai_cfg.get("temperature", 0)

        # 3. API key desde entorno
        self.api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "ANTHROPIC_API_KEY no está configurada. "
                "Las llamadas a la API fallarán."
            )

        # 4. Instanciar cliente de Anthropic
        self.client = Anthropic(api_key=self.api_key)

        logger.info(
            "ClaudeTrader inicializado — modelo=%s  max_tokens=%d  temperature=%.1f",
            self.model,
            self.max_tokens,
            self.temperature,
        )

    # ----------------------------------------------------------------
    # Método principal
    # ----------------------------------------------------------------
    def analyze_market_data(
        self,
        current_data: Union[Dict[str, Any], pd.DataFrame],
        context_docs: Optional[str] = None,
    ) -> str:
        """
        Envía los datos de mercado a Claude y devuelve la decisión limpia.

        Parameters
        ----------
        current_data : dict | pd.DataFrame
            Datos de la vela actual: precio, volumen, indicadores, etc.
            Si es un DataFrame se convierte a string para el prompt.
        context_docs : str, optional
            Texto adicional con contexto de estrategia o documentación
            de referencia que se adjuntará al prompt.

        Returns
        -------
        str
            ``"BUY"`` | ``"SELL"`` | ``"HOLD"``

        Raises
        ------
        RuntimeError
            Si la API key no está configurada.
        """
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY no está configurada. "
                "Revisa tu archivo .env o las variables de entorno."
            )

        # ── Construir el user prompt ────────────────────────────────
        user_prompt = self._build_user_prompt(current_data, context_docs)

        # ── Llamar a la API de Anthropic ────────────────────────────
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )

            # Extraer texto de la respuesta
            raw_response = message.content[0].text
            decision = raw_response.strip().upper()

            # Validar que sea una decisión esperada
            if decision not in ("BUY", "SELL", "HOLD"):
                logger.warning(
                    "Respuesta inesperada de Claude: '%s' — defaulting to HOLD",
                    raw_response,
                )
                return "HOLD"

            logger.info("ClaudeTrader decisión: %s", decision)
            return decision

        except Exception as e:
            logger.error("Error al llamar a la API de Anthropic: %s", str(e))
            raise

    # ----------------------------------------------------------------
    # Helpers privados
    # ----------------------------------------------------------------
    @staticmethod
    def _load_ai_config(config_path: str) -> Dict[str, Any]:
        """Lee la sección ``ai_engine`` del archivo YAML."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            ai_cfg = cfg.get("ai_engine", {})
            if not ai_cfg:
                logger.warning(
                    "Sección 'ai_engine' no encontrada en %s — usando defaults.",
                    config_path,
                )
            return ai_cfg
        except FileNotFoundError:
            logger.warning(
                "Archivo de configuración no encontrado: %s — usando defaults.",
                config_path,
            )
            return {}

    @staticmethod
    def _build_user_prompt(
        data: Union[Dict[str, Any], pd.DataFrame],
        context_docs: Optional[str] = None,
    ) -> str:
        """Convierte los datos de mercado en un prompt textual."""
        parts: list[str] = []

        # Contexto adicional (documentos de estrategia, etc.)
        if context_docs:
            parts.append(f"### Reference Context\n{context_docs}\n")

        # Datos de mercado
        parts.append("### Current Market Data")
        if isinstance(data, pd.DataFrame):
            parts.append(data.to_string())
        elif isinstance(data, dict):
            for key, value in data.items():
                parts.append(f"- {key}: {value}")
        else:
            parts.append(str(data))

        parts.append("\nBased on the data above, what is your trading decision?")
        return "\n".join(parts)
