<script lang="ts">
	import BentoCard from '$lib/components/BentoCard.svelte';
	import BacktestController from '$lib/components/BacktestController.svelte';
	import ConvictionRadar from '$lib/components/ConvictionRadar.svelte';
	import EngineMatrix from '$lib/components/EngineMatrix.svelte';
	import EquityChart from '$lib/components/EquityChart.svelte';
	import LogicExplainer from '$lib/components/LogicExplainer.svelte';
	import CalibrationWing from '$lib/components/CalibrationWing.svelte';
	import ForensicScrubber from '$lib/components/ForensicScrubber.svelte';
	import { appState } from '$lib/state/app.svelte';

	let clockTime = $state(new Date().toLocaleTimeString('es-MX', { hour12: false }));

	$effect(() => {
		const interval = setInterval(() => {
			clockTime = new Date().toLocaleTimeString('es-MX', { hour12: false });
		}, 1000);
		return () => clearInterval(interval);
	});

	let lastLogs = $derived(appState.logs.slice(0, 12));
</script>

<!-- Row 1: Conviction Radar (2×2) + Estado (1×1) + Simulation (1×2 from BacktestController) -->
<ConvictionRadar />

<BentoCard title="Estado" variant={appState.systemStatus === 'ERROR' ? 'veto' : 'default'}>
	<div class="space-y-3">
		<div class="flex items-center justify-between">
			<span class="text-[10px] uppercase text-[var(--color-null-text-dim)]">Motor</span>
			<span class="font-data text-xs font-semibold text-[var(--color-null-text)]">
				{appState.systemStatus}
			</span>
		</div>
		<div class="flex items-center justify-between">
			<span class="text-[10px] uppercase text-[var(--color-null-text-dim)]">Conexión</span>
			<span class="font-data text-xs font-semibold text-[var(--color-null-text)]">
				{appState.connectionStatus}
			</span>
		</div>
		<div class="flex items-center justify-between">
			<span class="text-[10px] uppercase text-[var(--color-null-text-dim)]">Reintentos</span>
			<span class="font-data text-xs font-semibold text-[var(--color-null-text)]">
				{appState.telemetry.retryCount}
			</span>
		</div>
		<div class="h-px w-full bg-[var(--color-null-border)]"></div>
		<div class="flex items-center justify-between">
			<span class="text-[10px] uppercase text-[var(--color-null-text-dim)]">Convicción</span>
			<span class="font-data text-xs font-bold text-[var(--color-null-accent)]">{appState.conviction}%</span>
		</div>
		<div class="flex items-center justify-between">
			<span class="text-[10px] uppercase text-[var(--color-null-text-dim)]">Hora</span>
			<span class="font-data text-xs text-[var(--color-null-accent)]">{clockTime}</span>
		</div>
	</div>
</BentoCard>

<BacktestController />

<!-- Row 2: Engine Matrix (2×1) + Equity Chart (2×2) -->
<EngineMatrix />

<EquityChart />

<!-- Row 3: Logic Explainer (2×1) + Terminal (2×1) -->
<LogicExplainer />

<BentoCard title="Terminal" colSpan={2}>
	<div class="max-h-56 space-y-1 overflow-y-auto">
		{#each lastLogs as log}
			<div class="flex items-start gap-2 font-data text-[10px]">
				<span class="shrink-0 text-[var(--color-null-text-ghost)]">
					{new Date(log.timestamp).toLocaleTimeString('es-MX', { hour12: false })}
				</span>
				<span class="shrink-0"
					class:text-[var(--color-null-accent)]={log.level === 'INFO'}
					class:text-amber-400={log.level === 'WARN'}
					class:text-[var(--color-null-veto)]={log.level === 'ERROR'}
				>
					[{log.level}]
				</span>
				<span class="text-[var(--color-null-text-dim)]">{log.source}</span>
				<span class="text-[var(--color-null-text)]">{log.message}</span>
			</div>
		{/each}
		{#if lastLogs.length === 0}
			<div class="flex h-12 items-center justify-center">
				<span class="text-[10px] text-[var(--color-null-text-ghost)]">
					&gt;_ esperando output...
				</span>
			</div>
		{/if}
	</div>
</BentoCard>

<!-- Row 4: Señales Activas (3 col) + Calibration Wing (1 col) -->
<BentoCard title="Señales Activas" colSpan={3}>
	<div class="grid grid-cols-2 gap-2 md:grid-cols-4">
		{#each appState.currentSignals.slice(0, 8) as signal}
			<div class="flex items-center justify-between rounded-md bg-[var(--color-null-black)] px-3 py-2">
				<div class="flex items-center gap-2">
					<span class="font-data text-[10px] font-bold"
						class:text-[var(--color-null-accent)]={signal.direction === 'LONG'}
						class:text-[var(--color-null-veto)]={signal.direction === 'SHORT'}
					>
						{signal.direction}
					</span>
					<span class="font-data text-xs text-[var(--color-null-text)]">{signal.symbol}</span>
				</div>
				<span class="font-data text-xs text-[var(--color-null-text-dim)]">
					{(signal.confidence * 100).toFixed(0)}%
				</span>
			</div>
		{:else}
			<div class="col-span-full flex h-12 items-center justify-center">
				<span class="text-[10px] uppercase tracking-[0.2em] text-[var(--color-null-text-ghost)]">
					ESCANEANDO MERCADO
				</span>
			</div>
		{/each}
	</div>
</BentoCard>

<CalibrationWing />

<!-- Scrubber sits fixed at the bottom -->
<ForensicScrubber />
