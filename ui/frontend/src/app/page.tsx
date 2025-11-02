'use client';

import { useState } from 'react';
import { DashboardOverview } from '@/components/dashboard/DashboardOverview';
import { ContainerGrid } from '@/components/containers/ContainerGrid';
import { ResourceMonitor } from '@/components/resources/ResourceMonitor';
import { CreateContainerModal } from '@/components/containers/CreateContainerModal';
import { PlusIcon } from '@heroicons/react/24/outline';

export default function HomePage() {
  const [showCreateModal, setShowCreateModal] = useState(false);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            ML Container Management
          </h1>
          <p className="mt-2 text-gray-600">
            Manage and monitor your machine learning training containers
          </p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn-primary btn-lg"
        >
          <PlusIcon className="h-5 w-5 mr-2" />
          New Container
        </button>
      </div>

      {/* Dashboard Overview */}
      <DashboardOverview />

      {/* Resource Monitor */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <ResourceMonitor />
        </div>
        <div className="space-y-6">
          {/* Quick Stats */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">
                Quick Stats
              </h3>
            </div>
            <div className="card-body">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Active Containers</span>
                  <span className="text-lg font-semibold text-gray-900">3</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Total Training Time</span>
                  <span className="text-lg font-semibold text-gray-900">24.5h</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Completed Sessions</span>
                  <span className="text-lg font-semibold text-gray-900">12</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Average Reward</span>
                  <span className="text-lg font-semibold text-gray-900">85.2</span>
                </div>
              </div>
            </div>
          </div>

          {/* System Status */}
          <div className="card">
            <div className="card-header">
              <h3 className="text-lg font-medium text-gray-900">
                System Status
              </h3>
            </div>
            <div className="card-body">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">API Status</span>
                  <span className="badge-success">Online</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">GPU Available</span>
                  <span className="badge-success">Yes</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Storage Space</span>
                  <span className="badge-warning">78% Used</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Memory Usage</span>
                  <span className="badge-primary">45% Used</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Container Grid */}
      <ContainerGrid />

      {/* Create Container Modal */}
      <CreateContainerModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
      />
    </div>
  );
}
