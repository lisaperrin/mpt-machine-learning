<script>
  import { onMount, onDestroy } from 'svelte';
  import { BarChart3, TrendingUp, Activity, Loader, X } from 'lucide-svelte';
  import EfficientFrontier from '../../lib/components/EfficientFrontier.svelte';
  import CorrelationHeatmap from '../../lib/components/CorrelationHeatmap.svelte';
  import { API_BASE } from '../../lib/config.js';

  const defaultAssets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'TLT', 'GLD'];

  let loading = false;
  let error = null;
  let frontierData = null;
  let correlationData = null;
  let factorData = null;
  let activeTab = 'frontier';
  let elapsedSeconds = 0;
  let elapsedTimer = null;
  let abortController = null;

  onDestroy(() => { cancelLoad(); });

  function cancelLoad() {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    if (elapsedTimer) {
      clearInterval(elapsedTimer);
      elapsedTimer = null;
    }
    loading = false;
    elapsedSeconds = 0;
  }

  async function loadData() {
    cancelLoad();

    try {
      loading = true;
      error = null;
      elapsedSeconds = 0;
      abortController = new AbortController();
      const signal = abortController.signal;

      elapsedTimer = setInterval(() => { elapsedSeconds += 1; }, 1000);

      const assets = defaultAssets.join(',');

      const [frontierResponse, correlationResponse, factorResponse] = await Promise.all([
        fetch(`${API_BASE}/api/efficient-frontier?assets=${assets}&num_portfolios=50`, { signal }),
        fetch(`${API_BASE}/api/correlations?assets=${assets}`, { signal }),
        fetch(`${API_BASE}/api/factor-exposures?assets=${assets}`, { signal })
      ]);

      if (frontierResponse.ok) {
        const data = await frontierResponse.json();
        if (data.success) frontierData = data;
      }

      if (correlationResponse.ok) {
        const data = await correlationResponse.json();
        if (data.success) correlationData = data;
      }

      if (factorResponse.ok) {
        const data = await factorResponse.json();
        if (data.success) factorData = data;
      }

    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Visualization error:', err);
      error = err.message;
    } finally {
      if (elapsedTimer) clearInterval(elapsedTimer);
      elapsedTimer = null;
      loading = false;
    }
  }

  function handleTabKeydown(event, tabs) {
    const currentIndex = tabs.indexOf(activeTab);
    let newIndex = currentIndex;
    if (event.key === 'ArrowRight') newIndex = (currentIndex + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') newIndex = (currentIndex - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') newIndex = 0;
    else if (event.key === 'End') newIndex = tabs.length - 1;
    else return;
    event.preventDefault();
    activeTab = tabs[newIndex];
    document.getElementById(`tab-${tabs[newIndex]}`)?.focus();
  }

  onMount(() => {
    loadData();
  });
</script>

<svelte:head>
  <title>Portfolio Visualization - PortfolioML</title>
</svelte:head>

<div class="space-y-6">
  <div class="flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <BarChart3 class="h-8 w-8 text-primary-600" />
      <div>
        <h1 class="text-3xl font-bold text-gray-900">Portfolio Visualization</h1>
        <p class="text-gray-600">Interactive charts and analysis</p>
      </div>
    </div>
    <div class="flex items-center space-x-3">
      {#if loading}
        <span class="text-sm text-gray-500">{elapsedSeconds}s</span>
        <button on:click={cancelLoad} class="btn btn-secondary flex items-center space-x-1 text-sm">
          <X class="h-3 w-3" />
          <span>Cancel</span>
        </button>
      {/if}
      <button
        on:click={loadData}
        disabled={loading}
        class="btn btn-primary flex items-center space-x-2"
      >
        {#if loading}
          <Loader class="h-4 w-4 animate-spin" />
        {:else}
          <TrendingUp class="h-4 w-4" />
        {/if}
        <span>{loading ? 'Loading...' : 'Refresh'}</span>
      </button>
    </div>
  </div>

  {#if error}
    <div class="card bg-red-50 border-red-200">
      <div class="flex items-center space-x-3">
        <Activity class="h-6 w-6 text-red-500" />
        <div>
          <h3 class="font-semibold text-red-800">Error Loading Data</h3>
          <p class="text-red-600 mt-1">{error}</p>
        </div>
      </div>
    </div>
  {:else}
    <div class="border-b border-gray-200">
      <nav class="flex space-x-8" role="tablist" aria-label="Visualization views">
        {#each [
          { id: 'frontier', label: 'Efficient Frontier' },
          { id: 'correlations', label: 'Correlations' },
          { id: 'factors', label: 'Factor Exposures' }
        ] as tab}
          <button
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls="tabpanel-{tab.id}"
            id="tab-{tab.id}"
            on:click={() => activeTab = tab.id}
            on:keydown={(e) => handleTabKeydown(e, ['frontier', 'correlations', 'factors'])}
            class="py-2 px-1 border-b-2 font-medium text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2
                   {activeTab === tab.id
                     ? 'border-primary-500 text-primary-600'
                     : 'border-transparent text-gray-500 hover:text-gray-700'}"
          >
            {tab.label}
          </button>
        {/each}
      </nav>
    </div>

    {#if loading}
      <div class="card">
        <div class="flex items-center justify-center py-12">
          <Loader class="h-8 w-8 animate-spin text-primary-600 mr-3" />
          <span class="text-lg text-gray-600">Loading visualization data...</span>
        </div>
      </div>

    {:else if activeTab === 'frontier'}
      <div role="tabpanel" id="tabpanel-frontier" aria-labelledby="tab-frontier" class="card">
        <div class="flex items-center space-x-3 mb-6">
          <TrendingUp class="h-6 w-6 text-blue-500" />
          <h2 class="text-xl font-semibold text-gray-900">Efficient Frontier</h2>
        </div>
        {#if frontierData}
          <EfficientFrontier frontierData={frontierData} />
        {:else}
          <p class="text-gray-500 text-center py-8">No frontier data available</p>
        {/if}
      </div>

    {:else if activeTab === 'correlations'}
      <div role="tabpanel" id="tabpanel-correlations" aria-labelledby="tab-correlations" class="card">
        <div class="flex items-center space-x-3 mb-6">
          <Activity class="h-6 w-6 text-blue-500" />
          <h2 class="text-xl font-semibold text-gray-900">Asset Correlation Matrix</h2>
        </div>
        {#if correlationData}
          <CorrelationHeatmap correlationData={correlationData} />
        {:else}
          <p class="text-gray-500 text-center py-8">No correlation data available</p>
        {/if}
      </div>

    {:else if activeTab === 'factors'}
      <div role="tabpanel" id="tabpanel-factors" aria-labelledby="tab-factors" class="card">
        <div class="flex items-center space-x-3 mb-6">
          <BarChart3 class="h-6 w-6 text-purple-500" />
          <h2 class="text-xl font-semibold text-gray-900">Factor Exposures</h2>
        </div>
        {#if factorData}
          <div class="mb-6">
            <h3 class="text-lg font-medium text-gray-800 mb-4">Portfolio-Level Exposures</h3>
            <div class="grid grid-cols-3 gap-4">
              <div class="bg-blue-50 p-4 rounded-lg">
                <p class="text-sm text-blue-600">Market Beta</p>
                <p class="text-2xl font-bold text-blue-800">{factorData.portfolio_exposures.beta.toFixed(2)}</p>
              </div>
              <div class="bg-green-50 p-4 rounded-lg">
                <p class="text-sm text-green-600">12M Momentum</p>
                <p class="text-2xl font-bold text-green-800">{(factorData.portfolio_exposures.momentum * 100).toFixed(1)}%</p>
              </div>
              <div class="bg-orange-50 p-4 rounded-lg">
                <p class="text-sm text-orange-600">Volatility</p>
                <p class="text-2xl font-bold text-orange-800">{(factorData.portfolio_exposures.volatility * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>

          <h3 class="text-lg font-medium text-gray-800 mb-4">Individual Asset Exposures</h3>
          <div class="overflow-x-auto">
            <table class="min-w-full text-sm">
              <thead>
                <tr class="border-b">
                  <th class="text-left py-2 px-3 font-semibold">Asset</th>
                  <th class="text-right py-2 px-3 font-semibold">Beta</th>
                  <th class="text-right py-2 px-3 font-semibold">12M Momentum</th>
                  <th class="text-right py-2 px-3 font-semibold">Volatility</th>
                  <th class="text-right py-2 px-3 font-semibold">Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {#each Object.entries(factorData.asset_exposures) as [asset, exposures]}
                  <tr class="border-b hover:bg-gray-50">
                    <td class="py-2 px-3 font-medium">{asset}</td>
                    <td class="text-right py-2 px-3">{exposures.beta.toFixed(2)}</td>
                    <td class="text-right py-2 px-3 {exposures.momentum_12m > 0 ? 'text-green-600' : 'text-red-600'}">
                      {(exposures.momentum_12m * 100).toFixed(1)}%
                    </td>
                    <td class="text-right py-2 px-3">{(exposures.volatility * 100).toFixed(1)}%</td>
                    <td class="text-right py-2 px-3 {exposures.sharpe_ratio > 0 ? 'text-green-600' : 'text-red-600'}">
                      {exposures.sharpe_ratio.toFixed(2)}
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {:else}
          <p class="text-gray-500 text-center py-8">No factor data available</p>
        {/if}
      </div>
    {/if}
  {/if}
</div>
