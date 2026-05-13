<script lang="ts">
	import BentoCard from './BentoCard.svelte';
	import { appState, getActiveEngines } from '$lib/state/app.svelte';
	import type { EngineState } from '$lib/state/app.svelte';

	// ── Tooltip State ──
	let activeTooltip = $state<string | null>(null);
	let tooltipX = $state(0);
	let tooltipY = $state(0);

	function showTooltip(engineId: string, event: MouseEvent | TouchEvent) {
		activeTooltip = engineId;
		if ('touches' in event) {
			tooltipX = event.touches[0].clientX;
			tooltipY = event.touches[0].clientY;
		} else {
			tooltipX = event.clientX;
			tooltipY = event.clientY;
		}
	}

	function hideTooltip() {
		activeTooltip = null;
	}

	// ── Helpers ──
	function getStatusColor(engine: EngineState): string {
		if (engine.status === 'ONLINE') return 'var(--color-null-accent)';
		if (engine.status === 'ERROR') return 'var(--color-null-veto)';
		return '#52525b'; // zinc-600
	}

	function getStatusBg(engine: EngineState): string {
		if (engine.status === 'OFFLINE') return 'grayscale(1) opacity-50';
		return '';
	}

	let activeTooltipEngine = $derived(
		activeTooltip ? getActiveEngines().find((e) => e.id === activeTooltip) : null
	);
</script>

<BentoCard title="Matriz de Motores" colSpan={2}>
	<!-- 2x5 Grid -->
	<div class="grid grid-cols-5 gap-2">
		{#each getActiveEngines() as engine (engine.id)}
			{@const isOffline = engine.status === 'OFFLINE'}
			{@const isError = engine.status === 'ERROR'}
			<button
				class="group relative flex flex-col items-center gap-1.5 rounded-lg border px-2 py-3
					transition-all duration-200 active:scale-95 cursor-pointer
					{isOffline
						? 'border-[var(--color-null-border)] bg-[var(--color-null-black)] grayscale'
						: isError
							? 'border-red-500/30 bg-red-500/5'
							: 'border-[var(--color-null-border)] bg-[var(--color-null-black)] hover:border-[var(--color-null-border-hover)] hover:bg-[var(--color-null-surface)]'
					}"
				onclick={(e) => showTooltip(engine.id, e)}
				ontouchstart={(e) => showTooltip(engine.id, e)}
				onmouseleave={hideTooltip}
				ontouchend={hideTooltip}
				aria-label="Motor {engine.name}"
			>
				<!-- Pulse Indicator -->
				<div class="relative">
					<div
						class="h-2 w-2 rounded-full transition-colors duration-200"
						style="background-color: {getStatusColor(engine)};"
					></div>
					{#if engine.pulseActive && engine.status === 'ONLINE'}
						<div
							class="absolute inset-0 h-2 w-2 rounded-full animate-ping"
							style="background-color: {getStatusColor(engine)}; opacity: 0.6;"
						></div>
					{/if}
					{#if engine.status === 'ONLINE'}
						<div
							class="absolute inset-0 h-2 w-2 rounded-full blur-sm"
							style="background-color: {getStatusColor(engine)}; opacity: 0.4;"
						></div>
					{/if}
				</div>

				<!-- Engine Name -->
				<span
					class="text-center text-[8px] font-medium uppercase leading-tight tracking-[0.1em]
						{isOffline ? 'text-zinc-600' : 'text-[var(--color-null-text-dim)]'}"
				>
					{engine.name}
				</span>

				<!-- Weight -->
				<span
					class="font-data text-[9px]
						{isOffline ? 'text-zinc-700' : 'text-[var(--color-null-text-ghost)]'}"
				>
					W: {engine.weight.toFixed(1)}
				</span>

				<!-- Offline/Error Badge -->
				{#if isOffline}
					<span class="absolute -top-1 -right-1 rounded-sm bg-zinc-700 px-1 py-0.5 text-[6px] font-bold uppercase text-zinc-400">
						OFF
					</span>
				{/if}
				{#if isError}
					<span class="absolute -top-1 -right-1 rounded-sm bg-red-500 px-1 py-0.5 text-[6px] font-bold uppercase text-white">
						ERR
					</span>
				{/if}
			</button>
		{/each}
	</div>

	<!-- Tooltip (positioned fixed, appears on touch/hover) -->
	{#if activeTooltipEngine}
		<div
			class="fixed z-[100] max-w-xs rounded-lg border border-[var(--color-null-border-hover)] bg-[var(--color-null-surface)] px-4 py-3 shadow-xl"
			style="left: {Math.min(tooltipX, window.innerWidth - 260)}px; top: {Math.max(tooltipY - 80, 10)}px; pointer-events: none;"
		>
			<div class="flex items-center gap-2 mb-2">
				<div
					class="h-2 w-2 rounded-full"
					style="background-color: {getStatusColor(activeTooltipEngine)};"
				></div>
				<span class="font-data text-[10px] font-bold uppercase tracking-[0.15em] text-[var(--color-null-text)]">
					{activeTooltipEngine.name}
				</span>
				<span class="font-data text-[9px] text-[var(--color-null-text-ghost)] ml-auto">
					{activeTooltipEngine.status}
				</span>
			</div>
			<div class="space-y-1">
				<div class="flex justify-between">
					<span class="text-[8px] uppercase text-[var(--color-null-text-ghost)]">Último Veredicto</span>
					<span class="font-data text-[9px] font-semibold text-[var(--color-null-accent)]">
						{activeTooltipEngine.lastVerdict}
					</span>
				</div>
				<div class="flex justify-between">
					<span class="text-[8px] uppercase text-[var(--color-null-text-ghost)]">Peso</span>
					<span class="font-data text-[9px] text-[var(--color-null-text)]">
						{activeTooltipEngine.weight.toFixed(2)}
					</span>
				</div>
				{#if activeTooltipEngine.lastUpdate > 0}
					<div class="flex justify-between">
						<span class="text-[8px] uppercase text-[var(--color-null-text-ghost)]">Última Señal</span>
						<span class="font-data text-[9px] text-[var(--color-null-text-dim)]">
							{new Date(activeTooltipEngine.lastUpdate).toLocaleTimeString('es-MX', { hour12: false })}
						</span>
					</div>
				{/if}
			</div>
		</div>
	{/if}
</BentoCard>
