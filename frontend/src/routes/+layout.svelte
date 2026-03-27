<script>
  import '../app.css';
  import { page } from '$app/stores';
  import { TrendingUp, BarChart3, Settings, History, Eye, Menu, X } from 'lucide-svelte';

  $: currentPath = $page.url.pathname;

  let mobileMenuOpen = false;

  $: if (currentPath) mobileMenuOpen = false;

  const navLinks = [
    { href: '/', label: 'Dashboard', icon: null },
    { href: '/optimize', label: 'Optimize', icon: Settings },
    { href: '/visualize', label: 'Visualize', icon: Eye },
    { href: '/backtest', label: 'Backtest', icon: History }
  ];
</script>

<div class="min-h-screen bg-gray-50">
  <!-- Navigation -->
  <nav class="bg-white border-b border-gray-200 sticky top-0 z-10" aria-label="Main navigation">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16">
        <div class="flex items-center space-x-8">
          <a href="/" class="flex items-center space-x-2">
            <TrendingUp class="h-8 w-8 text-primary-600" />
            <span class="font-bold text-xl text-gray-900">PortfolioML</span>
          </a>

          <div class="hidden md:flex space-x-6">
            {#each navLinks as link}
              <a
                href={link.href}
                class="px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center space-x-1
                       {currentPath === link.href
                         ? 'bg-primary-100 text-primary-700'
                         : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'}"
                aria-current={currentPath === link.href ? 'page' : undefined}
              >
                {#if link.icon}
                  <svelte:component this={link.icon} class="h-4 w-4" />
                {/if}
                <span>{link.label}</span>
              </a>
            {/each}
          </div>
        </div>

        <div class="flex items-center space-x-3">
          <span class="hidden sm:inline text-sm text-gray-500">ML-Enhanced Portfolio Optimization</span>
          <button
            class="md:hidden p-2 rounded-md text-gray-500 hover:text-gray-700 hover:bg-gray-100"
            on:click={() => mobileMenuOpen = !mobileMenuOpen}
            aria-expanded={mobileMenuOpen}
            aria-controls="mobile-menu"
            aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
          >
            {#if mobileMenuOpen}
              <X class="h-6 w-6" />
            {:else}
              <Menu class="h-6 w-6" />
            {/if}
          </button>
        </div>
      </div>
    </div>

    {#if mobileMenuOpen}
      <div id="mobile-menu" class="md:hidden border-t border-gray-200 bg-white">
        <div class="px-4 py-3 space-y-1">
          {#each navLinks as link}
            <a
              href={link.href}
              class="block px-3 py-2 rounded-md text-base font-medium transition-colors flex items-center space-x-2
                     {currentPath === link.href
                       ? 'bg-primary-100 text-primary-700'
                       : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'}"
              aria-current={currentPath === link.href ? 'page' : undefined}
            >
              {#if link.icon}
                <svelte:component this={link.icon} class="h-5 w-5" />
              {/if}
              <span>{link.label}</span>
            </a>
          {/each}
        </div>
      </div>
    {/if}
  </nav>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
    <slot />
  </main>
</div>