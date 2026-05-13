<script lang="ts">
	import { appState } from '$lib/state/app.svelte';

	type IgnitionState = 'IDLE' | 'LOADING' | 'ACTIVE' | 'ERROR';

	interface IgnitionSwitchProps {
		onToggle?: () => void;
	}

	let { onToggle = () => {} }: IgnitionSwitchProps = $props();

	// Map system + connection state to ignition visual state
	let ignitionState: IgnitionState = $derived(
		appState.systemStatus === 'ERROR'
			? 'ERROR'
			: appState.systemStatus === 'RUNNING'
				? 'ACTIVE'
				: appState.connectionStatus === 'CONNECTING' || appState.connectionStatus === 'RECONNECTING'
					? 'LOADING'
					: 'IDLE'
	);

	let isTransitioning = $state(false);
	let transitionTimeout: ReturnType<typeof setTimeout> | null = null;

	// Visual mappings per state
	let borderClass = $derived(
		ignitionState === 'ACTIVE'
			? 'border-[var(--color-null-accent)]/50'
			: ignitionState === 'ERROR'
				? 'border-red-500'
				: ignitionState === 'LOADING'
					? 'border-[var(--color-null-border-hover)]'
					: 'border-[var(--color-null-border)]'
	);

	let textClass = $derived(
		ignitionState === 'ACTIVE'
			? 'text-[var(--color-null-accent)]'
			: ignitionState === 'ERROR'
				? 'text-red-500'
				: ignitionState === 'LOADING'
					? 'text-[var(--color-null-text-dim)]'
					: 'text-[var(--color-null-text-ghost)]'
	);

	let label = $derived(
		isTransitioning
			? 'ESTABLECIENDO VÍNCULO...'
			: ignitionState === 'ACTIVE'
				? 'SISTEMA OPERATIVO // NY_SRV'
				: ignitionState === 'ERROR'
					? 'ERROR DE CONEXIÓN'
					: ignitionState === 'LOADING'
						? 'ESTABLECIENDO VÍNCULO...'
						: 'SISTEMA OFFLINE'
	);

	let glowStyle = $derived(
		ignitionState === 'ACTIVE'
			? 'box-shadow: 0 0 20px rgba(163, 230, 53, 0.2), 0 0 60px rgba(163, 230, 53, 0.05);'
			: ignitionState === 'ERROR'
				? 'box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);'
				: ''
	);

	let canInteract = $derived(
		!isTransitioning && ignitionState !== 'LOADING'
	);

	async function handlePress() {
		if (!canInteract) return;

		// Optimistic UI: immediately go to LOADING
		isTransitioning = true;

		// 10s timeout → auto-ERROR if no response
		transitionTimeout = setTimeout(() => {
			if (isTransitioning) {
				isTransitioning = false;
				appState.systemStatus = 'ERROR';
			}
		}, 10000);

		try {
			await onToggle?.();
		} catch {
			// Error handled in state manager
		} finally {
			isTransitioning = false;
			if (transitionTimeout) {
				clearTimeout(transitionTimeout);
				transitionTimeout = null;
			}
		}
	}
</script>

<button
	id="ignition-switch-btn"
	class="group relative flex w-full cursor-pointer items-center gap-4 rounded-[var(--radius-button)] border px-5 py-3.5
		transition-all duration-300
		active:scale-[0.97]
		disabled:cursor-not-allowed disabled:opacity-30
		{borderClass}"
	style="background: {ignitionState === 'LOADING' ? 'rgba(255,255,255,0.03)' : 'transparent'}; {glowStyle}"
	disabled={!canInteract}
	onclick={handlePress}
	aria-label="Interruptor maestro del sistema"
	aria-pressed={ignitionState === 'ACTIVE'}
>
	<!-- Animated border spinner for LOADING state -->
	{#if ignitionState === 'LOADING' || isTransitioning}
		<div class="absolute inset-0 overflow-hidden rounded-[var(--radius-button)]">
			<div
				class="absolute inset-[-2px] rounded-[var(--radius-button)]"
				style="
					background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.15), transparent);
					animation: orbit 2s linear infinite;
				"
			></div>
			<div class="absolute inset-[1px] rounded-[calc(var(--radius-button)-1px)] bg-[var(--color-null-surface)]"></div>
		</div>
	{/if}

	<!-- Vibration for ERROR state -->
	{#if ignitionState === 'ERROR'}
		<div class="contents" style="animation: vibrate 0.3s ease-in-out infinite;">
		</div>
	{/if}

	<!-- Indicator Dot -->
	<div class="relative z-10 flex items-center gap-3">
		<div class="relative">
			<div
				class="h-3 w-3 rounded-full transition-colors duration-300"
				class:bg-[var(--color-null-accent)]={ignitionState === 'ACTIVE'}
				class:bg-red-500={ignitionState === 'ERROR'}
				class:bg-zinc-600={ignitionState === 'IDLE'}
				class:bg-[var(--color-null-text-dim)]={ignitionState === 'LOADING' || isTransitioning}
			></div>
			{#if ignitionState === 'ACTIVE'}
				<div class="absolute inset-0 h-3 w-3 rounded-full bg-[var(--color-null-accent)] opacity-40 blur-sm"></div>
			{/if}
			{#if ignitionState === 'ERROR'}
				<div class="absolute inset-0 h-3 w-3 rounded-full bg-red-500 opacity-40 blur-sm"></div>
			{/if}
		</div>

		<!-- Label -->
		<div class="relative z-10 flex flex-col">
			<span
				class="font-data text-[10px] font-semibold uppercase tracking-[0.2em] transition-colors duration-300 {textClass}"
				class:animate-pulse={ignitionState === 'LOADING' || isTransitioning}
				style={ignitionState === 'ERROR' ? 'animation: vibrate 0.15s ease-in-out infinite;' : ''}
			>
				{label}
			</span>
			{#if ignitionState === 'ACTIVE'}
				<span class="font-data text-[8px] tracking-[0.1em] text-[var(--color-null-accent)]/50">
					LATENCIA: {appState.telemetry.latencyMs ?? '—'}ms
				</span>
			{/if}
		</div>
	</div>

	<!-- Right: Power icon -->
	<div class="relative z-10 ml-auto">
		<svg
			class="h-5 w-5 transition-colors duration-300 {textClass}"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			stroke-width="2"
			stroke-linecap="round"
			stroke-linejoin="round"
		>
			<path d="M18.36 6.64a9 9 0 1 1-12.73 0" />
			<line x1="12" y1="2" x2="12" y2="12" />
		</svg>
	</div>
</button>
