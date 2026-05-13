<script lang="ts">
	import BentoCard from './BentoCard.svelte';
	import { appState, updateConfig } from '$lib/state/app.svelte';

	let aiWeight = $state(appState.calibration.aiWeight);
	let smcSensitivity = $state(appState.calibration.smcSensitivity);
	let riskThreshold = $state(appState.calibration.riskThreshold);

	function handleChange() {
		updateConfig({
			aiWeight,
			smcSensitivity,
			riskThreshold
		});
	}
	
	// Keep local state in sync if backend updates it remotely
	$effect(() => {
		if (!appState.calibrationSyncing) {
			aiWeight = appState.calibration.aiWeight;
			smcSensitivity = appState.calibration.smcSensitivity;
			riskThreshold = appState.calibration.riskThreshold;
		}
	});
</script>

<BentoCard title="Calibraci&oacute;n" colSpan={1}>
	<div class="flex h-full flex-col justify-between space-y-4">
		<!-- Slider: PESO IA -->
		<div class="flex flex-col gap-1.5">
			<div class="flex items-center justify-between">
				<span class="text-[9px] uppercase tracking-widest text-[var(--color-null-text-dim)]">
					Peso IA
				</span>
				<span class="font-data text-[10px] font-bold text-[var(--color-null-accent)]">
					{aiWeight.toFixed(2)}
				</span>
			</div>
			<input
				type="range"
				min="0"
				max="2"
				step="0.05"
				bind:value={aiWeight}
				oninput={handleChange}
				class="slider"
				aria-label="Ajustar Peso IA"
			/>
		</div>

		<!-- Slider: SENSIBILIDAD SMC -->
		<div class="flex flex-col gap-1.5">
			<div class="flex items-center justify-between">
				<span class="text-[9px] uppercase tracking-widest text-[var(--color-null-text-dim)]">
					Sensibilidad SMC
				</span>
				<span class="font-data text-[10px] font-bold text-[var(--color-null-accent)]">
					{smcSensitivity.toFixed(2)}
				</span>
			</div>
			<input
				type="range"
				min="0"
				max="1"
				step="0.05"
				bind:value={smcSensitivity}
				oninput={handleChange}
				class="slider"
				aria-label="Ajustar Sensibilidad SMC"
			/>
		</div>

		<!-- Slider: UMBRAL DE RIESGO -->
		<div class="flex flex-col gap-1.5">
			<div class="flex items-center justify-between">
				<span class="text-[9px] uppercase tracking-widest text-[var(--color-null-text-dim)]">
					Umbral Riesgo
				</span>
				<span class="font-data text-[10px] font-bold text-[var(--color-null-veto)]">
					{riskThreshold.toFixed(2)}
				</span>
			</div>
			<input
				type="range"
				min="0.1"
				max="1"
				step="0.05"
				bind:value={riskThreshold}
				oninput={handleChange}
				class="slider slider-veto"
				aria-label="Ajustar Umbral de Riesgo"
			/>
		</div>

		<!-- Sync Status Feedback -->
		<div class="mt-auto flex h-6 items-center justify-center pt-2">
			{#if appState.calibrationSyncing}
				<div class="flex items-center gap-2 animate-pulse">
					<svg class="h-3 w-3 animate-spin text-[var(--color-null-accent)]" viewBox="0 0 24 24" fill="none" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
					</svg>
					<span class="text-[8px] font-bold uppercase tracking-[0.1em] text-[var(--color-null-accent)]">
						CAMBIO PENDIENTE DE SINCRONIZACI&Oacute;N
					</span>
				</div>
			{/if}
		</div>
	</div>
</BentoCard>

<style>
	/* Custom Tailwind-styled range sliders */
	.slider {
		-webkit-appearance: none;
		appearance: none;
		width: 100%;
		height: 3px;
		background: #27272a; /* zinc-800 */
		border-radius: 2px;
		outline: none;
		cursor: pointer;
	}

	.slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 12px;
		height: 12px;
		background: #ffffff; /* white puro */
		border-radius: 50%;
		border: 2px solid #000000;
		box-shadow: 0 0 4px rgba(255, 255, 255, 0.4);
		transition: transform 0.1s ease;
	}

	.slider::-webkit-slider-thumb:hover {
		transform: scale(1.3);
	}

	.slider::-webkit-slider-thumb:active {
		transform: scale(1.1);
	}

	.slider::-moz-range-thumb {
		width: 12px;
		height: 12px;
		background: #ffffff;
		border-radius: 50%;
		border: 2px solid #000000;
		box-shadow: 0 0 4px rgba(255, 255, 255, 0.4);
		transition: transform 0.1s ease;
	}

	.slider::-moz-range-thumb:hover {
		transform: scale(1.3);
	}

	/* Veto styled slider (Umbral de Riesgo) */
	.slider-veto::-webkit-slider-thumb {
		box-shadow: 0 0 4px rgba(249, 115, 22, 0.6);
	}
	.slider-veto::-moz-range-thumb {
		box-shadow: 0 0 4px rgba(249, 115, 22, 0.6);
	}
</style>
