<script>
  import { onMount, onDestroy } from 'svelte';
  import { TrendingUp, Brain, Shield, BarChart3, Activity, Zap } from 'lucide-svelte';
  import PerformanceChart from '../lib/components/PerformanceChart.svelte';
  import MetricCard from '../lib/components/MetricCard.svelte';
  import PortfolioTable from '../lib/components/PortfolioTable.svelte';
  import CorrelationHeatmap from '../lib/components/CorrelationHeatmap.svelte';
  import EfficientFrontier from '../lib/components/EfficientFrontier.svelte';
  import { API_BASE } from '../lib/config.js';
  import { formatPercent, formatNumber } from '../lib/utils/format.js';

  let loading = true;
  let error = null;
  let portfolioData = null;
  let correlationData = null;
  let frontierData = null;
  let activeTab = 'overview';
  let elapsedSeconds = 0;
  let elapsedTimer = null;
  let abortController = null;

  const defaultAssets = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "TLT", "GLD", "VNQ"];

  onMount(async () => {
    await loadDashboardData();
  });

  onDestroy(() => {
    cancelRequest();
  });

  function cancelRequest() {
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

  async function loadDashboardData() {
    cancelRequest();

    try {
      loading = true;
      error = null;
      elapsedSeconds = 0;
      abortController = new AbortController();
      const signal = abortController.signal;

      elapsedTimer = setInterval(() => { elapsedSeconds += 1; }, 1000);

      const [optimizeResponse, correlationResponse, frontierResponse] = await Promise.all([
        fetch(`${API_BASE}/api/optimize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            assets: defaultAssets,
            constraints: { min_weight: 0.05, max_weight: 0.30 }
          }),
          signal
        }),
        fetch(`${API_BASE}/api/correlations?assets=${defaultAssets.join(',')}`, { signal }),
        fetch(`${API_BASE}/api/efficient-frontier?assets=${defaultAssets.join(',')}`, { signal })
      ]);

      if (!optimizeResponse.ok) {
        throw new Error(`Optimization failed: ${optimizeResponse.status}`);
      }

      const data = await optimizeResponse.json();
      if (!data.success) {
        throw new Error(data.error || 'Optimization failed');
      }

      portfolioData = {
        strategies: data.results,
        assetCount: data.selected_assets.length,
        assets: data.selected_assets
      };

      if (correlationResponse.ok) {
        const corrData = await correlationResponse.json();
        if (corrData.success) {
          correlationData = corrData;
        }
      }

      if (frontierResponse.ok) {
        const frData = await frontierResponse.json();
        if (frData.success) {
          frontierData = frData;
        }
      }

    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Dashboard error:', err);
      error = err.message;
    } finally {
      if (elapsedTimer) clearInterval(elapsedTimer);
      elapsedTimer = null;
      loading = false;
    }
  }

  $: bestStrategy = getBestStrategy();

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

  function getBestStrategy() {
    if (!portfolioData?.strategies) return null;
    let best = null;
    let bestSharpe = -Infinity;
    for (const [name, data] of Object.entries(portfolioData.strategies)) {
      const sharpe = data.metrics?.Sharpe_Ratio || 0;
      if (sharpe > bestSharpe) {
        bestSharpe = sharpe;
        best = { name, sharpe, data };
      }
    }
    return best;
  }
</script>

<svelte:head>
  <title>Portfolio Dashboard - PortfolioML</title>
</svelte:head>

<div class="space-y-6">
  <div class="flex items-center justify-between">
    <div>
      <h1 class="text-3xl font-bold text-gray-900">Portfolio Dashboard</h1>
      <p class="text-gray-600 mt-2">ML-enhanced quantitative finance analytics</p>
    </div>
    <button
      on:click={loadDashboardData}
      disabled={loading}
      class="btn btn-primary flex items-center space-x-2"
    >
      <TrendingUp class="h-4 w-4" />
      <span>{loading ? 'Loading...' : 'Refresh Data'}</span>
    </button>
  </div>

  {#if loading}
    <div class="flex flex-col items-center justify-center py-12 space-y-4">
      <div class="flex items-center">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        <div class="ml-4">
          <span class="text-gray-700 font-medium">Computing optimal portfolios...</span>
          <p class="text-sm text-gray-500">{elapsedSeconds}s elapsed</p>
        </div>
      </div>
      <button on:click={cancelRequest} class="btn btn-secondary text-sm">
        Cancel
      </button>
    </div>
  {:else if error}
    <div class="card bg-red-50 border-red-200">
      <div class="flex items-center space-x-3">
        <Shield class="h-6 w-6 text-red-500" />
        <div>
          <h3 class="font-semibold text-red-800">Error Loading Dashboard</h3>
          <p class="text-red-600 mt-1">{error}</p>
        </div>
      </div>
    </div>
  {:else if portfolioData}
    <div class="border-b border-gray-200">
      <nav class="flex space-x-8" role="tablist" aria-label="Dashboard views">
        {#each [
          { id: 'overview', label: 'Overview' },
          { id: 'frontier', label: 'Efficient Frontier' },
          { id: 'correlations', label: 'Correlations' },
          { id: 'allocations', label: 'Allocations' }
        ] as tab}
          <button
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls="tabpanel-{tab.id}"
            id="tab-{tab.id}"
            on:click={() => activeTab = tab.id}
            on:keydown={(e) => handleTabKeydown(e, ['overview', 'frontier', 'correlations', 'allocations'])}
            class="py-2 px-1 border-b-2 font-medium text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2
                   {activeTab === tab.id
                     ? 'border-primary-500 text-primary-600'
                     : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}"
          >
            {tab.label}
          </button>
        {/each}
      </nav>
    </div>

    {#if activeTab === 'overview'}
      <div role="tabpanel" id="tabpanel-overview" aria-labelledby="tab-overview" class="grid grid-cols-1 md:grid-cols-5 gap-4">
        <MetricCard
          title="Equal Weight"
          value={formatPercent(portfolioData.strategies['Equal Weight']?.metrics?.Annualized_Return || 0)}
          subtitle="Annual Return"
          icon={BarChart3}
          color="gray"
        />
        <MetricCard
          title="Max Sharpe (MPT)"
          value={formatPercent(portfolioData.strategies['Max Sharpe (MPT)']?.metrics?.Annualized_Return || 0)}
          subtitle="Annual Return"
          icon={TrendingUp}
          color="blue"
        />
        <MetricCard
          title="Min Volatility"
          value={formatPercent(portfolioData.strategies['Min Volatility (MPT)']?.metrics?.Annualized_Return || 0)}
          subtitle="Annual Return"
          icon={Shield}
          color="green"
        />
        <MetricCard
          title="Factor-Based"
          value={formatPercent(portfolioData.strategies['Factor-Based']?.metrics?.Annualized_Return || 0)}
          subtitle="Annual Return"
          icon={Activity}
          color="blue"
        />
        <MetricCard
          title="Optimal ML"
          value={formatPercent(portfolioData.strategies['Optimal ML (HRP+Black-Litterman)']?.metrics?.Annualized_Return || 0)}
          subtitle="Annual Return"
          icon={Brain}
          color="purple"
          highlighted={true}
        />
      </div>

      <div class="card">
        <div class="flex items-center justify-between mb-6">
          <h2 class="text-xl font-semibold text-gray-900">Strategy Performance Comparison</h2>
          <span class="text-sm text-gray-500">{portfolioData.assetCount} assets</span>
        </div>
        <PerformanceChart strategies={portfolioData.strategies} />
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="card">
          <h3 class="text-lg font-semibold text-gray-900 mb-4">Risk Metrics</h3>
          <div class="space-y-3">
            {#each Object.entries(portfolioData.strategies).slice(0, 3) as [name, data]}
              <div class="flex justify-between items-center py-2 border-b border-gray-100">
                <span class="font-medium">{name}</span>
                <div class="flex space-x-4 text-sm">
                  <span class="text-gray-500">Vol: {formatPercent(data.metrics?.Annualized_Volatility || 0)}</span>
                  <span class="text-gray-500">Sharpe: {formatNumber(data.metrics?.Sharpe_Ratio || 0)}</span>
                  <span class="text-red-500">MaxDD: {formatPercent(Math.abs(data.metrics?.Max_Drawdown || 0))}</span>
                </div>
              </div>
            {/each}
          </div>
        </div>

        {#if bestStrategy}
          <div class="card bg-gradient-to-br from-purple-50 to-indigo-50">
            <div class="flex items-center space-x-2 mb-4">
              <Zap class="h-5 w-5 text-purple-600" />
              <h3 class="text-lg font-semibold text-purple-900">Best Strategy</h3>
            </div>
            <p class="text-2xl font-bold text-purple-800">{bestStrategy.name}</p>
            <div class="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div>
                <p class="text-purple-600">Sharpe Ratio</p>
                <p class="text-xl font-semibold">{formatNumber(bestStrategy.sharpe, 3)}</p>
              </div>
              <div>
                <p class="text-purple-600">Return</p>
                <p class="text-xl font-semibold">{formatPercent(bestStrategy.data.metrics?.Annualized_Return || 0)}</p>
              </div>
            </div>
          </div>
        {/if}
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
          <p class="text-gray-500 text-center py-8">Loading efficient frontier data...</p>
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
          <p class="text-gray-500 text-center py-8">Loading correlation data...</p>
        {/if}
      </div>

    {:else if activeTab === 'allocations'}
      <div role="tabpanel" id="tabpanel-allocations" aria-labelledby="tab-allocations" class="card">
        <h2 class="text-xl font-semibold text-gray-900 mb-6">Portfolio Allocations</h2>
        <PortfolioTable strategies={portfolioData.strategies} />
      </div>
    {/if}
  {/if}
</div>
