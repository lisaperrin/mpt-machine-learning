<script>
  import { onDestroy } from 'svelte';
  import { Settings, TrendingUp, Loader, X } from 'lucide-svelte';
  import PerformanceChart from '../../lib/components/PerformanceChart.svelte';
  import PortfolioTable from '../../lib/components/PortfolioTable.svelte';
  import { API_BASE } from '../../lib/config.js';

  const assetUniverse = {
    large_cap_tech: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX'],
    financials: ['JPM', 'V', 'MA', 'BAC', 'WFC', 'GS'],
    healthcare: ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT'],
    consumer_discretionary: ['HD', 'NKE', 'SBUX', 'MCD', 'LOW', 'DIS'],
    consumer_staples: ['PG', 'COST', 'WMT', 'KO', 'PEP', 'CVS'],
    industrials: ['BA', 'CAT', 'HON', 'UNP', 'UPS', 'FDX'],
    energy: ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'KMI'],
    bonds_and_commodities: ['TLT', 'IEF', 'LQD', 'GLD', 'SLV', 'VNQ']
  };

  const allAssets = [...new Set(Object.values(assetUniverse).flat())];

  let selectedAssets = new Set(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']);
  let constraints = {
    minWeight: 1.0,
    maxWeight: 25.0
  };

  let optimizing = false;
  let results = null;
  let error = null;
  let elapsedSeconds = 0;
  let elapsedTimer = null;
  let abortController = null;

  onDestroy(() => { cancelOptimization(); });

  function cancelOptimization() {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
    optimizing = false;
    elapsedSeconds = 0;
  }

  function toggleAsset(asset) {
    if (selectedAssets.has(asset)) {
      selectedAssets.delete(asset);
    } else {
      selectedAssets.add(asset);
    }
    selectedAssets = new Set(selectedAssets);
  }

  function selectAll() {
    selectedAssets = new Set(allAssets);
  }

  function clearAll() {
    selectedAssets = new Set();
  }

  async function runOptimization() {
    if (selectedAssets.size < 2) {
      error = 'Please select at least 2 assets for optimization';
      return;
    }

    cancelOptimization();

    try {
      optimizing = true;
      error = null;
      elapsedSeconds = 0;
      abortController = new AbortController();

      elapsedTimer = setInterval(() => { elapsedSeconds += 1; }, 1000);

      const response = await fetch(`${API_BASE}/api/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assets: Array.from(selectedAssets),
          constraints: {
            min_weight: constraints.minWeight / 100,
            max_weight: constraints.maxWeight / 100
          }
        }),
        signal: abortController.signal
      });

      if (!response.ok) {
        throw new Error(`Optimization failed: ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Optimization failed');
      }

      results = data.results;
    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Optimization error:', err);
      error = err.message;
    } finally {
      if (elapsedTimer) clearInterval(elapsedTimer);
      elapsedTimer = null;
      optimizing = false;
    }
  }
</script>

<svelte:head>
  <title>Portfolio Optimizer - PortfolioML</title>
</svelte:head>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <!-- Configuration Panel -->
  <div class="lg:col-span-1">
    <div class="card sticky top-20">
      <div class="flex items-center space-x-2 mb-6">
        <Settings class="h-6 w-6 text-gray-600" />
        <h2 class="text-xl font-semibold text-gray-900">Configuration</h2>
      </div>

      <!-- Asset Selection -->
      <div class="mb-6">
        <div class="flex items-center justify-between mb-3">
          <label class="text-sm font-medium text-gray-700">Asset Selection</label>
          <span class="text-xs text-gray-500">{selectedAssets.size} selected</span>
        </div>
        
        <div class="flex space-x-2 mb-3">
          <button 
            on:click={selectAll}
            class="btn btn-secondary text-xs flex-1"
          >
            Select All ({allAssets.length})
          </button>
          <button 
            on:click={clearAll}
            class="btn btn-secondary text-xs flex-1"
          >
            Clear All
          </button>
        </div>

        <div class="space-y-4 max-h-64 overflow-y-auto">
          {#each Object.entries(assetUniverse) as [sector, assets]}
            <div>
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                {sector.replace(/_/g, ' ')}
              </h4>
              <div class="grid grid-cols-3 gap-1">
                {#each assets as asset}
                  <label class="flex items-center">
                    <input 
                      type="checkbox" 
                      checked={selectedAssets.has(asset)}
                      on:change={() => toggleAsset(asset)}
                      class="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span class="ml-1 text-xs text-gray-700">{asset}</span>
                  </label>
                {/each}
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Constraints -->
      <div class="mb-6">
        <label class="text-sm font-medium text-gray-700 mb-3 block">Portfolio Constraints</label>
        <div class="space-y-4">
          <div>
            <label class="text-xs text-gray-600">Min Weight (%)</label>
            <input 
              type="number" 
              bind:value={constraints.minWeight}
              min="0" 
              max="10" 
              step="0.1"
              class="w-full mt-1 block px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
            />
          </div>
          <div>
            <label class="text-xs text-gray-600">Max Weight (%)</label>
            <input 
              type="number" 
              bind:value={constraints.maxWeight}
              min="5" 
              max="100" 
              step="1"
              class="w-full mt-1 block px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
            />
          </div>
        </div>
      </div>

      <!-- Optimize Button -->
      {#if optimizing}
        <div class="space-y-2">
          <div class="w-full bg-primary-50 border border-primary-200 rounded-md p-3">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <Loader class="h-4 w-4 animate-spin text-primary-600" />
                <span class="text-sm font-medium text-primary-700">Optimizing... {elapsedSeconds}s</span>
              </div>
              <button on:click={cancelOptimization} class="text-gray-400 hover:text-gray-600" title="Cancel">
                <X class="h-4 w-4" />
              </button>
            </div>
            {#if elapsedSeconds > 5}
              <p class="text-xs text-primary-500 mt-1">First run fetches market data and may take 30-60s</p>
            {/if}
          </div>
        </div>
      {:else}
        <button
          on:click={runOptimization}
          disabled={selectedAssets.size < 2}
          class="btn btn-primary w-full flex items-center justify-center space-x-2"
        >
          <TrendingUp class="h-4 w-4" />
          <span>Optimize Portfolio</span>
        </button>
      {/if}

      {#if error}
        <div class="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p class="text-sm text-red-600">{error}</p>
        </div>
      {/if}
    </div>
  </div>

  <!-- Results Panel -->
  <div class="lg:col-span-2">
    {#if results}
      <div class="space-y-6">
        <!-- Performance Chart -->
        <div class="card">
          <h3 class="text-xl font-semibold text-gray-900 mb-6">Strategy Performance</h3>
          <PerformanceChart strategies={results} />
        </div>

        <!-- Portfolio Table -->
        <div class="card">
          <h3 class="text-xl font-semibold text-gray-900 mb-6">Portfolio Allocations</h3>
          <PortfolioTable strategies={results} />
        </div>
      </div>
    {:else}
      <div class="card bg-gray-50 border-dashed">
        <div class="text-center py-12">
          <TrendingUp class="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 class="text-lg font-medium text-gray-900 mb-2">Portfolio Optimization</h3>
          <p class="text-gray-600">
            Select your assets and constraints, then click optimize to generate
            optimal portfolio allocations using Modern Portfolio Theory and machine learning.
          </p>
        </div>
      </div>
    {/if}
  </div>
</div>