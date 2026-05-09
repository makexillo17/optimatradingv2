"""
Feedback Loop Module — Sistema de Estado de Ánimo y Aprendizaje

Implementa:
1. Tracking de inactividad para definir el "Mood" (estado de ánimo).
2. Reducción dinámica de umbrales si el bot está 'BORED' (>48 velas sin operar).
3. Post-Mortem dinámico (Lessons Learned) después de cada trade cerrado.
"""

from typing import Dict, Any, Optional

class FeedbackLoop:
    def __init__(self, bored_threshold_candles: int = 48):
        self.bored_threshold = bored_threshold_candles
        self.candles_since_last_trade = 0
        self.mood = "FOCUSED"
        self.base_volume_threshold = 1.5
        self.current_volume_threshold = 1.5
        self.lessons_learned = []
        
    def update_mood(self) -> Dict[str, Any]:
        """Actualiza el estado de ánimo en base a la inactividad."""
        self.candles_since_last_trade += 1
        
        if self.candles_since_last_trade > self.bored_threshold:
            if self.mood != "BORED":
                print(f"[FeedbackLoop] Mood: BORED - Requesting Calibracion (Inactividad > {self.bored_threshold} velas)")
            self.mood = "BORED"
            # Bajar threshold temporalmente para buscar liquidez
            self.current_volume_threshold = max(1.2, self.base_volume_threshold * 0.8)  # 20% más permisivo
        else:
            self.mood = "FOCUSED"
            self.current_volume_threshold = self.base_volume_threshold
            
        return {
            'mood': self.mood,
            'candles_since_last_trade': self.candles_since_last_trade,
            'active_volume_threshold': self.current_volume_threshold
        }
        
    def reset_inactivity(self):
        """Reinicia el contador al abrir un trade."""
        self.candles_since_last_trade = 0
        if self.mood == "BORED":
            print("[FeedbackLoop] Mood: FOCUSED - Trade ejecutado, restaurando umbrales.")
        self.mood = "FOCUSED"
        self.current_volume_threshold = self.base_volume_threshold
        
    def generate_post_mortem(self, pnl: float, motor_culpable: str) -> str:
        """Genera una lección aprendida post-trade."""
        if pnl > 0:
            leccion = f"Reforzando estrategia por éxito en {motor_culpable}."
        else:
            leccion = f"Analizando error. Sugiriendo ajuste de stop loss para {motor_culpable}."
            
        self.lessons_learned.append({
            'pnl': pnl,
            'motor': motor_culpable,
            'lesson': leccion
        })
        return leccion
