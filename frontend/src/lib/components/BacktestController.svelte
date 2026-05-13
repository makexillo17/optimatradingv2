<script lang="ts">
	import BentoCard from './BentoCard.svelte';
	import { appState, runBacktest, addLog } from '$lib/state/app.svelte';

	// ── Simulation State ──
	let simulationActive = $derived(appState.simulationResults.status === 'RUNNING');
	let simulationDone = $derived(appState.simulationResults.status === 'COMPLETED');
	let simulationFailed = $derived(appState.simulationResults.status === 'FAILED');
	let abortController: AbortController | null = null;

	// ── Equity Curve ──
	let showEquityCurve = $state(false);
	let equityData = $derived(appState.simulationResults.equityCurve ?? []);
	let equityMax = $derived(equityData.length > 0 ? Math.max(...equityData) : 0);
	let equityMin = $derived(equityData.length > 0 ? Math.min(...equityData) : 0);
	let equityRange = $derived(equityMax - equityMin || 1);

	// SVG path for equity curve
	let equityPath = $derived.by(() => {
		if (equityData.length < 2) return '';
		const w = 100;
		const h = 60;
		const step = w / (equityData.length - 1);
		return equityData
			.map((val, i) => {
				const x = i * step;
				const y = h - ((val - equityMin) / equityRange) * h;
				return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
			})
			.join(' ');
	});

	// Trigger curve reveal on completion
	$effect(() => {
		if (simulationDone && equityData.length > 0) {
			// Delay for transition effect
			setTimeout(() => (showEquityCurve = true), 200);
		} else {
			showEquityCurve = false;
		}
	});

	// ── Actions ──
	async function handleRunSim() {
		if (simulationActive) return;
		showEquityCurve = false;
		abortController = new AbortController();

		try {
			await runBacktest();
		} catch (err) {
			const msg = (err as Error).message;
			if (msg.includes('SOURCE_DATA_NOT_FOUND') || msg.includes('btc_history')) {
				addLog('ERROR', '[FATAL] SOURCE_DATA_NOT_FOUND — btc_history.csv ausente', 'BACKTEST');
			}
		}
	}

	function handleAbort() {
		if (!simulationActive) return;
		abortController?.abort();
		abortController = null;
		appState.simulationResults = { status: 'FAILED' };
		addLog('WARN', 'Simulación abortada por usuario', 'BACKTEST');
	}
</script>

<BentoCard
	title="Simulación Interna"
	colSpan={2}
	variant={simulationFailed ? 'veto' : simulationDone ? 'accent' : 'default'}
>
	<!-- Controls Row -->
	<div class="mb-4 flex items-center gap-3">
		<!-- RUN SIM Button -->
		<button
			id="btn-run-sim"
			class="group relative flex items-center gap-2 rounded-[var(--radius-button)] border px-4 py-2
				font-data text-[10px] font-semibold uppercase tracking-[0.2em]
				transition-all duration-200 active:scale-95
				{simulationActive
					? 'border-[var(--color-null-border)] text-[var(--color-null-text-ghost)] cursor-not-allowed opacity-20'
					: 'border-[var(--color-null-accent)]/30 text-[var(--color-null-accent)] hover:border-[var(--color-null-accent)]/60 hover:bg-[var(--color-null-accent)]/5 cursor-pointer'
				}"
			disabled={simulationActive}
			onclick={handleRunSim}
		>
			{#if simulationActive}
				<div class="h-3 w-3 animate-spin rounded-full border border-transparent border-t-[var(--color-null-accent)]"></div>
			{:else}
				<svg class="h-3 w-3" viewBox="0 0 24 24" fill="currentColor">
					<polygon points="5,3 19,12 5,21" />
				</svg>
			{/if}
			RUN SIM
		</button>

		<!-- ABORT Button -->
		<button
			id="btn-abort-sim"
			class="group flex items-center gap-2 rounded-[var(--radius-button)] border px-4 py-2
				font-data text-[10px] font-semibold uppercase tracking-[0.2em]
				transition-all duration-200 active:scale-95
				{simulationActive
					? 'border-[var(--color-null-veto)]/30 text-[var(--color-null-veto)] hover:border-[var(--color-null-veto)]/60 hover:bg-[var(--color-null-veto)]/5 cursor-pointer'
					: 'border-[var(--color-null-border)] text-[var(--color-null-text-ghost)] cursor-not-allowed opacity-20'
				}"
			disabled={!simulationActive}
			onclick={handleAbort}
		>
			<svg class="h-3 w-3" viewBox="0 0 24 24" fill="currentColor">
				<rect x="6" y="6" width="12" height="12" rx="1" />
			</svg>
			ABORT
		</button>

		<!-- Status Badge -->
		{#if appState.simulationResults.status}
			<div class="ml-auto flex items-center gap-2">
				<div
					class="h-1.5 w-1.5 rounded-full"
					class:bg-[var(--color-null-accent)]={simulationDone}
					class:bg-[var(--color-null-veto)]={simulationFailed}
					class:bg-amber-400={simulationActive}
					class:animate-pulse={simulationActive}
				></div>
				<span
					class="font-data text-[9px] uppercase tracking-[0.15em]"
					class:text-[var(--color-null-accent)]={simulationDone}
					class:text-[var(--color-null-veto)]={simulationFailed}
					class:text-amber-400={simulationActive}
				>
					{appState.simulationResults.status}
				</span>
			</div>
		{/if}
	</div>

	<!-- Results Grid (visible after completion) -->
	{#if simulationDone && appState.simulationResults.totalReturn !== undefined}
		<div class="mb-4 grid grid-cols-4 gap-3">
			<div class="flex flex-col">
				<span class="text-[8px] uppercase text-[var(--color-null-text-ghost)]">Retorno</span>
				<span
					class="font-data text-sm font-bold"
					class:text-[var(--color-null-accent)]={(appState.simulationResults.totalReturn ?? 0) >= 0}
					class:text-[var(--color-null-veto)]={(appState.simulationResults.totalReturn ?? 0) < 0}
				>
					{appState.simulationResults.totalReturn?.toFixed(2)}%
				</span>
			</div>
			<div class="flex flex-col">
				<span class="text-[8px] uppercase text-[var(--color-null-text-ghost)]">Sharpe</span>
				<span class="font-data text-sm font-bold text-[var(--color-null-text)]">
					{appState.simulationResults.sharpeRatio?.toFixed(2) ?? '—'}
				</span>
			</div>
			<div class="flex flex-col">
				<span class="text-[8px] uppercase text-[var(--color-null-text-ghost)]">Drawdown</span>
				<span class="font-data text-sm font-bold text-[var(--color-null-veto)]">
					{appState.simulationResults.maxDrawdown?.toFixed(2) ?? '—'}%
				</span>
			</div>
			<div class="flex flex-col">
				<span class="text-[8px] uppercase text-[var(--color-null-text-ghost)]">Win Rate</span>
				<span class="font-data text-sm font-bold text-[var(--color-null-text)]">
					{appState.simulationResults.winRate?.toFixed(1) ?? '—'}%
				</span>
			</div>
		</div>
	{/if}

	<!-- Equity Curve SVG -->
	{#if showEquityCurve && equityData.length > 1}
		<div
			class="overflow-hidden rounded-md border border-[var(--color-null-border)] bg-[var(--color-null-black)] p-3
				transition-all duration-700 ease-out"
			style="opacity: {showEquityCurve ? 1 : 0}; transform: translateY({showEquityCurve ? 0 : 10}px);"
		>
			<div class="mb-2 flex items-center justify-between">
				<span class="text-[8px] uppercase tracking-[0.2em] text-[var(--color-null-text-ghost)]">
					EQUITY CURVE
				</span>
				<span class="font-data text-[8px] text-[var(--color-null-text-ghost)]">
					{equityData.length} candles
				</span>
			</div>
			<svg viewBox="0 0 100 60" class="h-24 w-full" preserveAspectRatio="none">
				<!-- Grid lines -->
				{#each [0, 15, 30, 45, 60] as y}
					<line x1="0" y1={y} x2="100" y2={y} stroke="rgba(255,255,255,0.05)" stroke-width="0.3" />
				{/each}
				<!-- Zero line -->
				<line
					x1="0"
					y1={60 - ((0 - equityMin) / equityRange) * 60}
					x2="100"
					y2={60 - ((0 - equityMin) / equityRange) * 60}
					stroke="rgba(255,255,255,0.1)"
					stroke-width="0.3"
					stroke-dasharray="2,2"
				/>
				<!-- Curve -->
				<path
					d={equityPath}
					fill="none"
					stroke="var(--color-null-accent)"
					stroke-width="1"
					stroke-linecap="round"
					stroke-linejoin="round"
					vector-effect="non-scaling-stroke"
				/>
				<!-- Glow behind curve -->
				<path
					d={equityPath}
					fill="none"
					stroke="var(--color-null-accent)"
					stroke-width="3"
					stroke-linecap="round"
					opacity="0.15"
					filter="blur(2px)"
					vector-effect="non-scaling-stroke"
				/>
			</svg>
		</div>
	{:else if !simulationDone && !simulationActive}
		<div class="flex h-20 items-center justify-center rounded-md border border-dashed border-[var(--color-null-border)] bg-[var(--color-null-black)]">
			<span class="text-[10px] uppercase tracking-[0.2em] text-[var(--color-null-text-ghost)]">
				{simulationFailed ? '⚠ SIMULACIÓN FALLIDA' : '▸ EJECUTAR SIMULACIÓN PARA VISUALIZAR'}
			</span>
		</div>
	{:else if simulationActive}
		<div class="flex h-20 items-center justify-center rounded-md border border-[var(--color-null-border)] bg-[var(--color-null-black)]">
			<div class="flex items-center gap-3">
				<div class="h-4 w-4 animate-spin rounded-full border-2 border-transparent border-t-[var(--color-null-accent)]"></div>
				<span class="font-data text-[10px] uppercase tracking-[0.15em] text-[var(--color-null-text-dim)] animate-pulse">
					PROCESANDO BACKTEST...
				</span>
			</div>
		</div>
	{/if}
</BentoCard>
