<script lang="ts">
	type SystemStatus = 'IDLE' | 'RUNNING' | 'ERROR' | 'RECONNECTING';

	interface StatusOrbitProps {
		status?: SystemStatus;
		label?: string;
	}

	let { status = 'IDLE', label = '' }: StatusOrbitProps = $props();

	let dotColor = $derived(
		status === 'RUNNING'
			? 'bg-[var(--color-null-accent)]'
			: status === 'ERROR'
				? 'bg-[var(--color-null-veto)]'
				: status === 'RECONNECTING'
					? 'bg-amber-400'
					: 'bg-zinc-600'
	);

	let pulseAnimation = $derived(
		status === 'RUNNING'
			? 'animate-[pulse-accent_2s_ease-in-out_infinite]'
			: status === 'ERROR'
				? 'animate-[pulse-veto_1.5s_ease-in-out_infinite]'
				: status === 'RECONNECTING'
					? 'animate-[pulse-veto_1s_ease-in-out_infinite]'
					: ''
	);

	let statusText = $derived(
		label ||
			(status === 'RUNNING'
				? 'CONECTADO'
				: status === 'ERROR'
					? 'ERROR'
					: status === 'RECONNECTING'
						? 'REINTENTANDO...'
						: 'INACTIVO')
	);

	let textColor = $derived(
		status === 'RUNNING'
			? 'text-[var(--color-null-accent)]'
			: status === 'ERROR'
				? 'text-[var(--color-null-veto)]'
				: status === 'RECONNECTING'
					? 'text-amber-400'
					: 'text-[var(--color-null-text-dim)]'
	);

	let orbitActive = $derived(status === 'RUNNING' || status === 'RECONNECTING');
</script>

<div class="flex items-center gap-3" id="status-orbit">
	<!-- Orbit Container -->
	<div class="relative flex h-8 w-8 items-center justify-center">
		<!-- Orbital Ring -->
		{#if orbitActive}
			<div
				class="absolute inset-0 rounded-full border border-[var(--color-null-border-hover)]"
				style="animation: orbit 4s linear infinite;"
			>
				<div
					class="absolute -top-[2px] left-1/2 h-1 w-1 -translate-x-1/2 rounded-full {dotColor}"
				></div>
			</div>
		{/if}

		<!-- Center Dot -->
		<div class="relative">
			<div class="h-2.5 w-2.5 rounded-full {dotColor} {pulseAnimation}"></div>
			{#if status === 'RUNNING'}
				<div
					class="absolute inset-0 h-2.5 w-2.5 rounded-full {dotColor} opacity-40 blur-sm"
				></div>
			{/if}
		</div>
	</div>

	<!-- Status Label -->
	<div class="flex flex-col">
		<span class="font-data text-[10px] font-semibold tracking-[0.15em] {textColor}">
			{statusText}
		</span>
		{#if status === 'RECONNECTING'}
			<span class="font-data text-[8px] text-[var(--color-null-text-ghost)]">
				BACKOFF ACTIVO
			</span>
		{/if}
	</div>
</div>
