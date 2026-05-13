<script lang="ts">
	import BentoCard from './BentoCard.svelte';
	import { appState } from '$lib/state/app.svelte';
	import type { TradeDetail } from '$lib/state/app.svelte';

	let detail = $derived(appState.selectedTradeDetail);
	let hasSelection = $derived(appState.selectedTradeIndex !== null);
	let selectedTrade = $derived(
		appState.selectedTradeIndex !== null && appState.simulationResults.trades
			? appState.simulationResults.trades.find((t) => t.index === appState.selectedTradeIndex)
			: null
	);

	let animKey = $state(0);
	$effect(() => {
		if (appState.selectedTradeIndex !== null) {
			animKey++;
		}
	});

	let displayDetail: TradeDetail = $derived(
		detail ?? {
			gapDetected: '\u2014', fibLevel: '\u2014', volumeFilter: '\u2014',
			smcSignal: '\u2014', liquidityZone: '\u2014', volatilityRegime: '\u2014',
			claudeVerdict: '\u2014', claudeConfidence: 0,
			claudeReasoning: 'Selecciona un trade en la gr\u00e1fica para ver el an\u00e1lisis completo.',
			positionSize: '\u2014', kellyFraction: '\u2014', stopLoss: '\u2014',
			takeProfit: '\u2014', riskRewardRatio: '\u2014', atrMultiple: '\u2014'
		}
	);

	let isBullish = $derived(
		displayDetail.claudeVerdict === 'BULLISH' || displayDetail.claudeVerdict === 'LONG'
	);
	let isBearish = $derived(
		displayDetail.claudeVerdict === 'BEARISH' || displayDetail.claudeVerdict === 'SHORT'
	);
	let verdictClass = $derived(
		isBullish
			? 'border-[rgba(163,230,53,0.3)] text-[var(--color-null-accent)]'
			: isBearish
				? 'border-[rgba(249,115,22,0.3)] text-[var(--color-null-veto)]'
				: 'border-zinc-600 text-zinc-400'
	);
	let tradeBadgeClass = $derived(
		selectedTrade?.result === 'WIN'
			? 'bg-[rgba(163,230,53,0.1)] text-[var(--color-null-accent)]'
			: 'bg-[rgba(249,115,22,0.1)] text-[var(--color-null-veto)]'
	);
</script>

<BentoCard title="Explicación Lógica" colSpan={2}>
	{#if hasSelection}
		{#if selectedTrade}
			<div class="mb-4 flex items-center gap-3 fade-section" style="--delay: 0ms;">
				<span class="rounded-sm px-2 py-0.5 font-data text-[10px] font-bold uppercase {tradeBadgeClass}">
					{selectedTrade.direction} {selectedTrade.result}
				</span>
				<span class="font-data text-xs text-[var(--color-null-text)]">
					{selectedTrade.symbol}
				</span>
				<span
					class="ml-auto font-data text-sm font-bold"
					class:text-[var(--color-null-accent)]={selectedTrade.pnl >= 0}
					class:text-[var(--color-null-veto)]={selectedTrade.pnl < 0}
				>
					{selectedTrade.pnl >= 0 ? '+' : ''}{selectedTrade.pnl.toFixed(2)}%
				</span>
			</div>
		{/if}

		{#key animKey}
			<div class="space-y-3">
				<div class="fade-section rounded-lg border border-[var(--color-null-border)] bg-[var(--color-null-black)] p-3" style="--delay: 50ms;">
					<h4 class="mb-2 text-[8px] font-bold uppercase tracking-[0.25em] text-[var(--color-null-text-ghost)]">
						&#9656; INPUTS T&Eacute;CNICOS
					</h4>
					<div class="grid grid-cols-3 gap-x-4 gap-y-2">
						{#each [
							['Gap Detectado', displayDetail.gapDetected, 'text-[var(--color-null-accent)]'],
							['FIB Level', displayDetail.fibLevel, 'text-[var(--color-null-text)]'],
							['Volume Filter', displayDetail.volumeFilter, displayDetail.volumeFilter === 'PASS' ? 'text-[var(--color-null-accent)]' : displayDetail.volumeFilter === 'FAIL' ? 'text-[var(--color-null-veto)]' : 'text-[var(--color-null-text)]'],
							['SMC Signal', displayDetail.smcSignal, 'text-[var(--color-null-text)]'],
							['Liquidity Zone', displayDetail.liquidityZone, 'text-[var(--color-null-text)]'],
							['Volatility', displayDetail.volatilityRegime, 'text-[var(--color-null-text)]']
						] as [label, value, colorClass]}
							<div class="flex flex-col">
								<span class="text-[7px] uppercase text-[var(--color-null-text-ghost)]">{label}</span>
								<code class="font-data text-[10px] {colorClass}">{value}</code>
							</div>
						{/each}
					</div>
				</div>

				<div class="fade-section rounded-lg border border-[var(--color-null-border)] bg-white/[0.03] p-3" style="--delay: 150ms;">
					<h4 class="mb-2 text-[8px] font-bold uppercase tracking-[0.25em] text-[var(--color-null-text-ghost)]">
						&#9656; VERDICTO CLAUDE
					</h4>
					<div class="mb-2 flex items-center gap-3">
						<span class="rounded-md border px-2.5 py-1 font-data text-xs font-bold uppercase tracking-[0.1em] {verdictClass}">
							{displayDetail.claudeVerdict}
						</span>
						{#if displayDetail.claudeConfidence > 0}
							<span class="font-data text-[10px] text-[var(--color-null-text-dim)]">
								Confianza: {displayDetail.claudeConfidence.toFixed(0)}%
							</span>
						{/if}
					</div>
					<p class="font-data text-[10px] leading-relaxed text-[var(--color-null-text-dim)] italic">
						"{displayDetail.claudeReasoning}"
					</p>
				</div>

				<div class="fade-section rounded-lg border border-[var(--color-null-border)] bg-[var(--color-null-black)] p-3" style="--delay: 250ms;">
					<h4 class="mb-2 text-[8px] font-bold uppercase tracking-[0.25em] text-[var(--color-null-text-ghost)]">
						&#9656; C&Aacute;LCULO DE RIESGO
					</h4>
					<div class="grid grid-cols-3 gap-x-4 gap-y-2">
						{#each [
							['Position Size', displayDetail.positionSize, 'text-[var(--color-null-text)]'],
							['Kelly f*', displayDetail.kellyFraction, 'text-[var(--color-null-text)]'],
							['ATR \u00d7', displayDetail.atrMultiple, 'text-[var(--color-null-text)]'],
							['Stop Loss', displayDetail.stopLoss, 'text-[var(--color-null-veto)]'],
							['Take Profit', displayDetail.takeProfit, 'text-[var(--color-null-accent)]'],
							['R:R Ratio', displayDetail.riskRewardRatio, 'text-[var(--color-null-text)]']
						] as [label, value, colorClass]}
							<div class="flex flex-col">
								<span class="text-[7px] uppercase text-[var(--color-null-text-ghost)]">{label}</span>
								<code class="font-data text-[10px] {colorClass}">{value}</code>
							</div>
						{/each}
					</div>
				</div>
			</div>
		{/key}
	{:else}
		<div class="flex h-40 flex-col items-center justify-center gap-3">
			<svg class="h-10 w-10 text-[var(--color-null-text-ghost)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
				<circle cx="12" cy="12" r="10" />
				<path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
				<line x1="12" y1="17" x2="12.01" y2="17" />
			</svg>
			<span class="text-[10px] uppercase tracking-[0.2em] text-[var(--color-null-text-ghost)]">
				SELECCIONA UN TRADE EN LA CURVA
			</span>
			<span class="text-[8px] text-[var(--color-null-text-ghost)]">
				Inputs, veredicto IA y c&aacute;lculo de riesgo aparecer&aacute;n aqu&iacute;
			</span>
		</div>
	{/if}
</BentoCard>

<style>
	.fade-section {
		animation: fadeSlideIn 0.4s ease-out both;
		animation-delay: var(--delay, 0ms);
	}
	@keyframes fadeSlideIn {
		from { opacity: 0; transform: translateY(6px); }
		to { opacity: 1; transform: translateY(0); }
	}
</style>
