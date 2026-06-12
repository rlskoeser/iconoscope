<script>
  import ScatterView from './ScatterView.svelte'
  import { clusters, hiddenClusters, searchQuery, showClusterOutlines } from './stores.js'

  function toggle(id) {
    hiddenClusters.update(s => {
      const n = new Set(s)
      n.has(id) ? n.delete(id) : n.add(id)
      return n
    })
  }
</script>

<div class="layout">
  <ScatterView />
  <aside>
    <input type="search" placeholder="Filter by filename…" bind:value={$searchQuery} />
    {#if $clusters.length}
      <section>
        <h4>Clusters</h4>
        <label class="toggle">
          <input type="checkbox" bind:checked={$showClusterOutlines} />
          Show outlines
        </label>
        {#each $clusters as c}
          <button class="cluster-btn" class:off={$hiddenClusters.has(c.id)} on:click={() => toggle(c.id)}>
            <span class="dot" style="background:{c.color}"></span>
            {c.label}
          </button>
        {/each}
      </section>
    {/if}
  </aside>
</div>

<style>
  .layout { display: flex; height: 100vh; background: #333; color: #ddd; font: 13px/1.5 system-ui, sans-serif; }
  aside { width: 200px; padding: 12px; background: #1c1c1c; display: flex; flex-direction: column; gap: 10px; border-left: 1px solid #333; overflow-y: auto; }
  input { width: 100%; padding: 5px 8px; background: #2a2a2a; border: 1px solid #444; color: #ddd; border-radius: 4px; font-size: 13px; }
  section { display: flex; flex-direction: column; gap: 4px; }
  h4 { margin: 0 0 4px; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: #666; }
  .toggle { display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 12px; color: #999; padding: 2px 0; }
  .toggle:hover { color: #ccc; }
  .cluster-btn { display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 3px 0; background: none; border: none; color: inherit; font: inherit; text-align: left; width: 100%; }
  .cluster-btn:hover { color: #fff; }
  .cluster-btn.off { opacity: 0.35; }
  .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
</style>
