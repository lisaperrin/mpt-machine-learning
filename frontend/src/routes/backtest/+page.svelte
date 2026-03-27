<script>
  import { onMount, onDestroy } from 'svelte';
  import { History, Play, TrendingUp, AlertTriangle, X } from 'lucide-svelte';
  import EquityCurve from '../../lib/components/EquityCurve.svelte';
  import DrawdownChart from '../../lib/components/DrawdownChart.svelte';
  import MonteCarloChart from '../../lib/components/MonteCarloChart.svelte';
  import { API_BASE } from '../../lib/config.js';

  const assetUniverse = {
    large_cap_tech: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
    financials: ['JPM', 'V', 'MA', 'BAC'],
    healthcare: ['JNJ', 'UNH', 'PFE'],
    bonds_commodities: ['TLT', 'GLD', 'VNQ']
  };

  let selectedAssets = new Set(['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'TLT', 'GLD', 'VNQ']);
  let rebalanceDays = 21;
  let loading = false;
  let error = null;
  let elapsedSeconds = 0;
  let elapsedTimer = null;
  let abortController = null;

  let backtestData = null;
  let riskData = null;
  let monteCarloData = null;

  let activeTab = 'equity';

  onDestroy(() => { cancelBacktest(); });

  function cancelBacktest() {
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

  function toggleAsset(asset) {
    if (selectedAssets.has(asset)) {
      selectedAssets.delete(asset);
    } else {
      selectedAssets.add(asset);
    }
    selectedAssets = new Set(selectedAssets);
  }

  async function runBacktest() {
    if (selectedAssets.size < 3) {
      error = 'Please select at least 3 assets';
      return;
    }

    cancelBacktest();

    try {
      loading = true;
      error = null;
      elapsedSeconds = 0;
      abortController = new AbortController();
      const signal = abortController.signal;

      elapsedTimer = setInterval(() => { elapsedSeconds += 1; }, 1000);

      const assets = Array.from(selectedAssets).join(',');

      const [backtestResponse, riskResponse, mcResponse] = await Promise.all([
        fetch(`${API_BASE}/api/backtest-detailed?assets=${assets}&rebalance_days=${rebalanceDays}`, { signal }),
        fetch(`${API_BASE}/api/risk-analysis?assets=${assets}&strategy=equal`, { signal }),
        fetch(`${API_BASE}/api/monte-carlo?assets=${assets}&num_simulations=1000&horizon_days=252`, { signal })
      ]);

      if (backtestResponse.ok) {
        const data = await backtestResponse.json();
        if (data.success) {
          backtestData = data;
        }
      }

      if (riskResponse.ok) {
        const data = await riskResponse.json();
        if (data.success) {
          riskData = data;
        }
      }

      if (mcResponse.ok) {
        const data = await mcResponse.json();
        if (data.success) {
          monteCarloData = data;
        }
      }

    } catch (err) {
      if (err.name === 'AbortError') return;
      console.error('Backtest error:', err);
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
    runBacktest();
  });
</script>

<svelte:head>
  <title>Backtesting - PortfolioML</title>
</svelte:head>

<div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
  <div class="lg:col-span-1">
    <div class="card sticky top-20">
      <div class="flex items-center space-x-2 mb-6">
        <History class="h-6 w-6 text-gray-600" />
        <h2 class="text-xl font-semibold text-gray-900">Backtest Config</h2>
      </div>

      <div class="mb-6">
        <label class="text-sm font-medium text-gray-700 mb-3 block">Assets ({selectedAssets.size})</label>
        <div class="space-y-3 max-h-48 overflow-y-auto">
          {#each Object.entries(assetUniverse) as [sector, assets]}
            <div>
              <h4 class="text-xs font-medium text-gray-500 uppercase mb-1">{sector.replace(/_/g, ' ')}</h4>
              <div class="flex flex-wrap gap-1">
                {#each assets as asset}
                  <button
                    on:click={() => toggleAsset(asset)}
                    class="px-2 py-1 text-xs rounded transition-colors
                           {selectedAssets.has(asset)
                             ? 'bg-primary-100 text-primary-700 border border-primary-300'
                             : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'}"
                  >
                    {asset}
                  </button>
                {/each}
              </div>
            </div>
          {/each}
        </div>
      </div>

      <div class="mb-6">
        <label class="text-sm font-medium text-gray-700 mb-2 block">Rebalance Frequency</label>
        <select
          bind:value={rebalanceDays}
          class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-primary-500 focus:border-primary-500"
        >
          <option value={5}>Weekly (5 days)</option>
          <option value={21}>Monthly (21 days)</option>
          <option value={63}>Quarterly (63 days)</option>
        </select>
      </div>

      {#if loading}
        <div class="w-full bg-primary-50 border border-primary-200 rounded-md p-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600"></div>
              <span class="text-sm font-medium text-primary-700">Running... {elapsedSeconds}s</span>
            </div>
            <button on:click={cancelBacktest} class="text-gray-400 hover:text-gray-600" title="Cancel">
              <X class="h-4 w-4" />
            </button>
          </div>
          {#if elapsedSeconds > 5}
            <p class="text-xs text-primary-500 mt-1">Backtests with many assets can take 1-2 minutes</p>
          {/if}
        </div>
      {:else}
        <button
          on:click={runBacktest}
          disabled={selectedAssets.size < 3}
          class="btn btn-primary w-full flex items-center justify-center space-x-2"
        >
          <Play class="h-4 w-4" />
          <span>Run Backtest</span>
        </button>
      {/if}

      {#if error}
        <div class="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p class="text-sm text-red-600">{error}</p>
        </div>
      {/if}
    </div>
  </div>

  <div class="lg:col-span-3">
    {#if backtestData || riskData || monteCarloData}
      <div class="border-b border-gray-200 mb-6">
        <nav class="flex space-x-8" role="tablist" aria-label="Backtest views">
          {#each [
            { id: 'equity', label: 'Equity Curves' },
            { id: 'drawdown', label: 'Risk Analysis' },
            { id: 'montecarlo', label: 'Monte Carlo' }
          ] as tab}
            <button
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls="tabpanel-{tab.id}"
              id="tab-{tab.id}"
              on:click={() => activeTab = tab.id}
              on:keydown={(e) => handleTabKeydown(e, ['equity', 'drawdown', 'montecarlo'])}
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

      {#if activeTab === 'equity' && backtestData}
        <div role="tabpanel" id="tabpanel-equity" aria-labelledby="tab-equity" class="card">
          <div class="flex items-center space-x-3 mb-6">
            <TrendingUp class="h-6 w-6 text-blue-500" />
            <h2 class="text-xl font-semibold text-gray-900">Strategy Equity Curves</h2>
          </div>
          <EquityCurve backtestData={backtestData} />
        </div>

      {:else if activeTab === 'drawdown' && riskData}
        <div role="tabpanel" id="tabpanel-drawdown" aria-labelledby="tab-drawdown" class="card">
          <div class="flex items-center space-x-3 mb-6">
            <AlertTriangle class="h-6 w-6 text-red-500" />
            <h2 class="text-xl font-semibold text-gray-900">Drawdown Analysis</h2>
          </div>
          <DrawdownChart riskData={riskData} />
        </div>

        <div class="card mt-6">
          <h3 class="text-lg font-semibold text-gray-900 mb-4">Risk Statistics</h3>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="bg-gray-50 p-4 rounded-lg">
              <p class="text-sm text-gray-500">Sortino Ratio</p>
              <p class="text-xl font-semibold">{riskData.risk_metrics?.Sortino_Ratio?.toFixed(3) || 'N/A'}</p>
            </div>
            <div class="bg-gray-50 p-4 rounded-lg">
              <p class="text-sm text-gray-500">Tail Ratio</p>
              <p class="text-xl font-semibold">{riskData.risk_metrics?.Tail_Ratio?.toFixed(3) || 'N/A'}</p>
            </div>
            <div class="bg-gray-50 p-4 rounded-lg">
              <p class="text-sm text-gray-500">Skewness</p>
              <p class="text-xl font-semibold">{riskData.risk_metrics?.Skewness?.toFixed(3) || 'N/A'}</p>
            </div>
            <div class="bg-gray-50 p-4 rounded-lg">
              <p class="text-sm text-gray-500">Kurtosis</p>
              <p class="text-xl font-semibold">{riskData.risk_metrics?.Kurtosis?.toFixed(3) || 'N/A'}</p>
            </div>
          </div>
        </div>

      {:else if activeTab === 'montecarlo' && monteCarloData}
        <div role="tabpanel" id="tabpanel-montecarlo" aria-labelledby="tab-montecarlo" class="card">
          <div class="flex items-center space-x-3 mb-6">
            <TrendingUp class="h-6 w-6 text-purple-500" />
            <h2 class="text-xl font-semibold text-gray-900">Monte Carlo Simulation</h2>
          </div>
          <MonteCarloChart monteCarloData={monteCarloData} />
        </div>

        <div class="card mt-6">
          <h3 class="text-lg font-semibold text-gray-900 mb-4">Terminal Value Distribution</h3>
          <div class="grid grid-cols-5 gap-4 text-sm">
            <div class="bg-red-50 p-3 rounded-lg text-center">
              <p class="text-red-600">5th %ile</p>
              <p class="text-lg font-semibold">{((monteCarloData.final_values.p5 - 1) * 100).toFixed(1)}%</p>
            </div>
            <div class="bg-orange-50 p-3 rounded-lg text-center">
              <p class="text-orange-600">25th %ile</p>
              <p class="text-lg font-semibold">{((monteCarloData.final_values.p25 - 1) * 100).toFixed(1)}%</p>
            </div>
            <div class="bg-blue-50 p-3 rounded-lg text-center">
              <p class="text-blue-600">Median</p>
              <p class="text-lg font-semibold">{((monteCarloData.final_values.p50 - 1) * 100).toFixed(1)}%</p>
            </div>
            <div class="bg-green-50 p-3 rounded-lg text-center">
              <p class="text-green-600">75th %ile</p>
              <p class="text-lg font-semibold">{((monteCarloData.final_values.p75 - 1) * 100).toFixed(1)}%</p>
            </div>
            <div class="bg-emerald-50 p-3 rounded-lg text-center">
              <p class="text-emerald-600">95th %ile</p>
              <p class="text-lg font-semibold">{((monteCarloData.final_values.p95 - 1) * 100).toFixed(1)}%</p>
            </div>
          </div>
        </div>
      {/if}

    {:else}
      <div class="card bg-gray-50 border-dashed">
        <div class="text-center py-12">
          <History class="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 class="text-lg font-medium text-gray-900 mb-2">Historical Backtesting</h3>
          <p class="text-gray-600">
            Select assets and click "Run Backtest" to see historical performance,
            drawdown analysis, and Monte Carlo simulations.
          </p>
        </div>
      </div>
    {/if}
  </div>
</div>
