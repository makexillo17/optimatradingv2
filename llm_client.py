"""
llm_client.py — ClaudeTrader
─────────────────────────────────────────────────────────────────────
Cliente LLM que conecta con la API de Anthropic (Claude) para
generar decisiones de trading cuantitativo: BUY | SELL | HOLD.

La configuración del modelo se lee de  config/config.yaml  (sección ai_engine)
y la API key se obtiene de la variable de entorno  ANTHROPIC_API_KEY.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

# ── Diagnóstico de entorno ──────────────────────────────────────────
print(f"[llm_client] Python ejecutable: {sys.executable}")

import yaml
import pandas as pd
from dotenv import load_dotenv

# ── Importación segura de anthropic ─────────────────────────────────
try:
    from anthropic import Anthropic
except ModuleNotFoundError:
    print(
        "\n[ERROR FATAL] No se encontró el módulo 'anthropic'.\n"
        "Instálalo con:  pip install anthropic\n"
        f"Python en uso:  {sys.executable}\n"
        "Si usas venv, asegúrate de activarlo primero.\n"
    )
    sys.exit(1)

logger = logging.getLogger("ClaudeTrader")

# ── Ruta absoluta al proyecto ───────────────────────────────────────
_PROJECT_ROOT = Path(r"c:\Users\chump\OneDrive\proyecto personal")

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

# Longitud esperada de una API key de Anthropic
_EXPECTED_KEY_LENGTH = 108


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
            config_path = str(_PROJECT_ROOT / "config" / "config.yaml")

        # 2. Leer configuración YAML
        ai_cfg = self._load_ai_config(config_path)
        self.model: str = ai_cfg.get("model", "claude-3-5-sonnet-20240620")
        self.max_tokens: int = ai_cfg.get("max_tokens", 1024)
        self.temperature: float = ai_cfg.get("temperature", 0)

        # 3. Cargar .env con ruta absoluta
        env_path = _PROJECT_ROOT / ".env"
        print(f"[ClaudeTrader] Cargando .env desde: {env_path}  (existe: {env_path.exists()})")

        dotenv_ok = load_dotenv(dotenv_path=str(env_path), override=True)
        print(f"[ClaudeTrader] load_dotenv() retorno: {dotenv_ok}")

        # 4. Leer API key — intento 1: desde os.environ (vía dotenv)
        raw_key = os.environ.get("ANTHROPIC_API_KEY", "")

        # 4b. Fallback: lectura manual del archivo .env linea por linea
        if not raw_key and env_path.exists():
            print("[ClaudeTrader] dotenv no cargo la variable. Intentando lectura manual...")
            try:
                with open(str(env_path), "r", encoding="utf-8-sig") as f:
                    for line_num, raw_line in enumerate(f, 1):
                        line = raw_line.strip()
                        # Mostrar cada linea para debug
                        print(f"  .env linea {line_num}: {repr(line)}")
                        # Ignorar comentarios y lineas vacias
                        if not line or line.startswith("#"):
                            continue
                        # Comparar en mayusculas para robustez
                        if line.upper().startswith("ANTHROPIC_API_KEY"):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                value = parts[1].strip().strip('"').strip("'")
                                if value:
                                    raw_key = value
                                    os.environ["ANTHROPIC_API_KEY"] = raw_key
                                    print(f"  Variable cargada: {raw_key[:10]}...")
                                else:
                                    print(f"  [!] Se encontro ANTHROPIC_API_KEY pero el valor esta VACIO.")
                                    print(f"  [!] Abre el .env y pega tu llave DESPUES del signo =")
                            break
            except Exception as e:
                print(f"[ClaudeTrader] Error leyendo .env manualmente: {e}")

        self.api_key: str = raw_key.strip()

        # 5. Validar API key
        if not self.api_key:
            print(
                "\n[ERROR FATAL] ANTHROPIC_API_KEY esta vacia.\n"
                "Abre el archivo .env y pega tu llave de Anthropic.\n"
                f"Ruta esperada: {env_path}\n"
            )
            sys.exit(1)

        if len(self.api_key) != _EXPECTED_KEY_LENGTH:
            print(
                f"\n[ERROR FATAL] ANTHROPIC_API_KEY tiene {len(self.api_key)} caracteres "
                f"(se esperaban {_EXPECTED_KEY_LENGTH}).\n"
                f"Primeros 5: {self.api_key[:5]}   Ultimos 4: {self.api_key[-4:]}\n"
                "Verifica que no haya espacios, comillas o saltos de linea extra.\n"
            )
            sys.exit(1)

        print(f"[ClaudeTrader] API Key validada: {self.api_key[:5]}...{self.api_key[-4:]}  (len={len(self.api_key)})")

        # 5. Instanciar cliente de Anthropic
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
