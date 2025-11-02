'use client';

import { ReactNode } from 'react';
import { SWRConfig } from 'swr';
import { WebSocketProvider } from '@/contexts/WebSocketContext';
import { fetcher } from '@/lib/api';

interface ProvidersProps {
  children: ReactNode;
}

export function Providers({ children }: ProvidersProps) {
  return (
    <SWRConfig
      value={{
        fetcher,
        refreshInterval: 30000, // Refresh every 30 seconds
        revalidateOnFocus: true,
        revalidateOnReconnect: true,
        errorRetryCount: 3,
        errorRetryInterval: 5000,
        onError: (error) => {
          console.error('SWR Error:', error);
        },
      }}
    >
      <WebSocketProvider>
        {children}
      </WebSocketProvider>
    </SWRConfig>
  );
}
