import { writable, derived } from 'svelte/store'

export const allPoints = writable(window.__DATA__?.points ?? [])
export const clusters = writable(window.__DATA__?.clusters ?? [])
export const hiddenClusters = writable(new Set())
export const showClusterOutlines = writable(true)
export const searchQuery = writable('')

export const visibleIds = derived(
  [allPoints, hiddenClusters, searchQuery],
  ([$all, $hidden, $q]) => {
    const q = $q.toLowerCase()
    return new Set(
      $all
        .filter(p => !$hidden.has(p.cluster) && (!q || p.path.toLowerCase().includes(q)))
        .map(p => p.id)
    )
  }
)
