<script lang="ts">
	import type { Snippet } from 'svelte';

	interface BentoCardProps {
		title?: string;
		colSpan?: number;
		rowSpan?: number;
		variant?: 'default' | 'accent' | 'veto';
		children: Snippet;
		headerRight?: Snippet;
	}

	let {
		title = '',
		colSpan = 1,
		rowSpan = 1,
		variant = 'default',
		children,
		headerRight
	}: BentoCardProps = $props();

	let isHovered = $state(false);
	let mouseX = $state(50);
	let mouseY = $state(50);

	let borderClass = $derived(
		isHovered ? 'border-[var(--color-null-border-hover)]' : 'border-[var(--color-null-border)]'
	);

	let glowClass = $derived(
		variant === 'accent'
			? 'shadow-[0_0_30px_var(--color-null-glow-accent)]'
			: variant === 'veto'
				? 'shadow-[0_0_30px_var(--color-null-glow-veto)]'
				: ''
	);

	function handleMouseMove(e: MouseEvent) {
		const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
		mouseX = ((e.clientX - rect.left) / rect.width) * 100;
		mouseY = ((e.clientY - rect.top) / rect.height) * 100;
	}
</script>

<div
	class="bento-card {borderClass} {glowClass}"
	style="grid-column: span {colSpan}; grid-row: span {rowSpan}; --mouse-x: {mouseX}%; --mouse-y: {mouseY}%;"
	role="region"
	aria-label={title || 'Módulo'}
	onmouseenter={() => (isHovered = true)}
	onmouseleave={() => (isHovered = false)}
	onmousemove={handleMouseMove}
>
	{#if title || headerRight}
		<div class="mb-3 flex items-center justify-between">
			{#if title}
				<h3 class="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-null-text-dim)]">
					{title}
				</h3>
			{/if}
			{#if headerRight}
				{@render headerRight()}
			{/if}
		</div>
	{/if}

	<div class="relative z-10">
		{@render children()}
	</div>
</div>
