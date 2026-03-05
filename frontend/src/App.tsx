import '@mantine/core/styles.css';

import { MantineProvider } from '@mantine/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import { AppShell, Group, NavLink, Title } from '@mantine/core';
import HomePage from './pages/HomePage';
import ProcessPage from './pages/ProcessPage';
import ResultsPage from './pages/ResultsPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MantineProvider>
        <BrowserRouter>
          <AppShell header={{ height: 60 }} padding="md">
            <AppShell.Header>
              <Group h="100%" px="md" justify="space-between">
                <Title order={3}>LRSLibrary</Title>
                <Group>
                  <NavLink component={Link} to="/" label="Home" />
                  <NavLink component={Link} to="/process" label="Process" />
                  <NavLink component={Link} to="/results" label="Results" />
                </Group>
              </Group>
            </AppShell.Header>
            <AppShell.Main>
              <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/process" element={<ProcessPage />} />
                <Route path="/results" element={<ResultsPage />} />
              </Routes>
            </AppShell.Main>
          </AppShell>
        </BrowserRouter>
      </MantineProvider>
    </QueryClientProvider>
  );
}
