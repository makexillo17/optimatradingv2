import os
from typing import Dict, Any, List
import openai
from utils.logger import setup_logger

class AnalysisExplainer:
    def __init__(self, api_key: str = None):
        self.logger = setup_logger("AnalysisExplainer")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key no encontrada")
            
        openai.api_key = self.api_key
        
    def explain_analysis(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera explicaciones en lenguaje natural del análisis
        
        Args:
            analysis_result: Resultado completo del análisis
            
        Returns:
            Dict con diferentes niveles de explicación
        """
        try:
            # Generar diferentes niveles de explicación
            summary = self._generate_summary(analysis_result)
            detailed = self._generate_detailed_explanation(analysis_result)
            technical = self._generate_technical_explanation(analysis_result)
            
            return {
                'summary': summary,
                'detailed': detailed,
                'technical': technical
            }
            
        except Exception as e:
            self.logger.error(f"Error generando explicación: {str(e)}")
            return {
                'summary': "Error generando explicación",
                'detailed': str(e),
                'technical': ""
            }
            
    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """Genera un resumen conciso del análisis"""
        prompt = self._create_summary_prompt(result)
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Eres un analista financiero experto que explica "
                          "recomendaciones de trading de forma clara y concisa "
                          "para inversores no técnicos."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    def _generate_detailed_explanation(self, result: Dict[str, Any]) -> str:
        """Genera una explicación detallada del análisis"""
        prompt = self._create_detailed_prompt(result)
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Eres un analista financiero experto que explica "
                          "estrategias de trading de forma detallada pero "
                          "comprensible para inversores intermedios."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    def _generate_technical_explanation(self, result: Dict[str, Any]) -> str:
        """Genera una explicación técnica detallada"""
        prompt = self._create_technical_prompt(result)
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Eres un analista cuantitativo experto que explica "
                          "estrategias de trading de forma técnica y detallada "
                          "para profesionales del mercado."
            }, {
                "role": "user",
                "content": prompt
            }],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    def _create_summary_prompt(self, result: Dict[str, Any]) -> str:
        """Crea el prompt para el resumen"""
        return f"""
        Genera un resumen breve y claro de la siguiente recomendación de trading:
        
        Activo: {result['asset_symbol']}
        Recomendación: {result['recommendation']}
        Confianza: {result['confidence']:.1%}
        
        Justificación original: {result['justification']}
        
        El resumen debe ser comprensible para inversores no técnicos y enfocarse en:
        1. La recomendación principal
        2. El nivel de confianza en términos simples
        3. Las razones principales en lenguaje sencillo
        """
        
    def _create_detailed_prompt(self, result: Dict[str, Any]) -> str:
        """Crea el prompt para la explicación detallada"""
        # Extraer los 3 módulos más influyentes
        top_modules = self._get_top_modules(result)
        module_details = "\n".join([
            f"- {module}: {result['module_results'][module]['justification']}"
            for module in top_modules
        ])
        
        return f"""
        Genera una explicación detallada de la siguiente recomendación de trading:
        
        Activo: {result['asset_symbol']}
        Recomendación: {result['recommendation']}
        Confianza: {result['confidence']:.1%}
        
        Módulos más influyentes:
        {module_details}
        
        Justificación general: {result['justification']}
        
        La explicación debe:
        1. Describir la recomendación y su contexto
        2. Explicar los factores principales que llevaron a esta decisión
        3. Discutir el nivel de confianza y sus implicaciones
        4. Mencionar posibles riesgos o consideraciones importantes
        """
        
    def _create_technical_prompt(self, result: Dict[str, Any]) -> str:
        """Crea el prompt para la explicación técnica"""
        module_details = "\n".join([
            f"- {module}: {details['justification']}"
            for module, details in result['module_results'].items()
        ])
        
        consensus_details = result.get('consensus_details', {})
        
        return f"""
        Genera una explicación técnica detallada de la siguiente recomendación de trading:
        
        Activo: {result['asset_symbol']}
        Recomendación: {result['recommendation']}
        Confianza: {result['confidence']:.1%}
        
        Resultados de módulos:
        {module_details}
        
        Detalles de consenso:
        {consensus_details}
        
        La explicación debe incluir:
        1. Análisis técnico detallado de cada factor relevante
        2. Discusión de las correlaciones y pesos entre módulos
        3. Análisis de la calidad y confiabilidad de las señales
        4. Consideraciones sobre la implementación de la estrategia
        5. Análisis de riesgos y factores de mercado relevantes
        """
        
    def _get_top_modules(self, result: Dict[str, Any], n: int = 3) -> List[str]:
        """Obtiene los n módulos más influyentes"""
        if 'consensus_details' not in result:
            return list(result['module_results'].keys())[:n]
            
        weights = result['consensus_details'].get('dynamic_weights', {})
        if not weights:
            return list(result['module_results'].keys())[:n]
            
        sorted_modules = sorted(
            weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [module for module, _ in sorted_modules[:n]] 