'use client';

import { Fragment, useState, useEffect } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { toast } from 'react-hot-toast';
import { containerAPI, trainingAPI, GameConfig } from '@/lib/api';

const createContainerSchema = z.object({
    name: z.string().min(1, 'Name is required'),
    config: z.object({
        game: z.string().min(1, 'Game is required'),
        algorithm: z.string().default('ppo'),
        total_timesteps: z.number().min(1000),
        vec_envs: z.number().min(1).max(16),
        learning_rate: z.number().min(1e-6).max(1e-1),
        checkpoint_every_sec: z.number().min(10).max(3600),
        video_recording: z.boolean(),
        fast_mode: z.boolean().default(false),
        resource_limits: z.object({
            cpu_cores: z.number().min(0.5).max(16),
            memory_gb: z.number().min(0.5).max(64),
            gpu_memory_gb: z.number().optional(),
            disk_space_gb: z.number().min(1),
        }),
    }),
});

type CreateContainerForm = z.infer<typeof createContainerSchema>;

interface CreateContainerModalProps {
    isOpen: boolean;
    onClose: () => void;
}

export function CreateContainerModal({ isOpen, onClose }: CreateContainerModalProps) {
    const [games, setGames] = useState<GameConfig[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [loadingGames, setLoadingGames] = useState(false);

    const { register, handleSubmit, watch, setValue, reset, formState: { errors } } = useForm<CreateContainerForm>({
        resolver: zodResolver(createContainerSchema),
        defaultValues: {
            config: {
                algorithm: 'ppo',
                total_timesteps: 1000000,
                vec_envs: 4,
                learning_rate: 2.5e-4,
                checkpoint_every_sec: 60,
                video_recording: true,
                fast_mode: false,
                resource_limits: {
                    cpu_cores: 2,
                    memory_gb: 4,
                    disk_space_gb: 10,
                },
            },
        },
    });

    const selectedGame = watch('config.game');
    const fastMode = watch('config.fast_mode');

    // Load supported games
    useEffect(() => {
        if (isOpen) {
            setLoadingGames(true);
            trainingAPI.games()
                .then(setGames)
                .catch((err) => {
                    console.error('Failed to load games:', err);
                    toast.error('Failed to load supported games');
                })
                .finally(() => setLoadingGames(false));
        }
    }, [isOpen]);

    // Update defaults when game changes
    useEffect(() => {
        const gameConfig = games.find(g => g.game === selectedGame);
        if (gameConfig) {
            // Only update if not already modified? For now just set defaults
            // We won't overwrite other fields to allow user persistence, but could suggest defaults
        }
    }, [selectedGame, games]);

    const onSubmit = async (data: CreateContainerForm) => {
        setIsLoading(true);
        try {
            await containerAPI.create(data);
            toast.success('Container created successfully');
            reset();
            onClose();
        } catch (error: any) {
            console.error('Failed to create container:', error);
            toast.error(error.response?.data?.detail || 'Failed to create container');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Transition.Root show={isOpen} as={Fragment}>
            <Dialog as="div" className="relative z-50" onClose={onClose}>
                <Transition.Child
                    as={Fragment}
                    enter="ease-out duration-300"
                    enterFrom="opacity-0"
                    enterTo="opacity-100"
                    leave="ease-in duration-200"
                    leaveFrom="opacity-100"
                    leaveTo="opacity-0"
                >
                    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
                </Transition.Child>

                <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
                    <div className="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
                        <Transition.Child
                            as={Fragment}
                            enter="ease-out duration-300"
                            enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                            enterTo="opacity-100 translate-y-0 sm:scale-100"
                            leave="ease-in duration-200"
                            leaveFrom="opacity-100 translate-y-0 sm:scale-100"
                            leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
                        >
                            <Dialog.Panel className="relative transform overflow-hidden rounded-lg bg-white px-4 pb-4 pt-5 text-left shadow-xl transition-all sm:my-8 sm:w-full sm:max-w-2xl sm:p-6">
                                <div className="absolute right-0 top-0 hidden pr-4 pt-4 sm:block">
                                    <button
                                        type="button"
                                        className="rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
                                        onClick={onClose}
                                    >
                                        <span className="sr-only">Close</span>
                                        <XMarkIcon className="h-6 w-6" aria-hidden="true" />
                                    </button>
                                </div>

                                <div className="sm:flex sm:items-start">
                                    <div className="mt-3 text-center sm:mt-0 sm:text-left w-full">
                                        <Dialog.Title as="h3" className="text-base font-semibold leading-6 text-gray-900">
                                            Create New Training Container
                                        </Dialog.Title>
                                        <div className="mt-6">
                                            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                                                {/* Basic Info */}
                                                <div className="grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-6">
                                                    <div className="sm:col-span-3">
                                                        <label className="block text-sm font-medium leading-6 text-gray-900">
                                                            Container Name
                                                        </label>
                                                        <div className="mt-2">
                                                            <input
                                                                type="text"
                                                                {...register('name')}
                                                                className="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                            />
                                                            {errors.name && (
                                                                <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>
                                                            )}
                                                        </div>
                                                    </div>

                                                    <div className="sm:col-span-3">
                                                        <label className="block text-sm font-medium leading-6 text-gray-900">
                                                            Game
                                                        </label>
                                                        <div className="mt-2">
                                                            {loadingGames ? (
                                                                <div className="text-sm text-gray-500">Loading games...</div>
                                                            ) : (
                                                                <select
                                                                    {...register('config.game')}
                                                                    className="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                                >
                                                                    <option value="">Select a game</option>
                                                                    {games.map((g) => (
                                                                        <option key={g.game} value={g.game}>
                                                                            {g.display_name}
                                                                        </option>
                                                                    ))}
                                                                </select>
                                                            )}
                                                            {errors.config?.game && (
                                                                <p className="mt-1 text-sm text-red-600">{errors.config.game.message}</p>
                                                            )}
                                                        </div>
                                                    </div>

                                                    <div className="sm:col-span-3">
                                                        <div className="flex items-start">
                                                            <div className="flex h-6 items-center">
                                                                <input
                                                                    id="fast-mode"
                                                                    type="checkbox"
                                                                    {...register('config.fast_mode')}
                                                                    className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-600"
                                                                />
                                                            </div>
                                                            <div className="ml-3">
                                                                <label htmlFor="fast-mode" className="text-sm font-medium leading-6 text-gray-900">
                                                                    Fast Mode ⚡
                                                                </label>
                                                                <p className="text-sm text-gray-500">
                                                                    Train without video for max speed, then auto-render after completion.
                                                                </p>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    {!fastMode && (
                                                        <div className="sm:col-span-3">
                                                            <div className="flex items-start">
                                                                <div className="flex h-6 items-center">
                                                                    <input
                                                                        id="video-recording"
                                                                        type="checkbox"
                                                                        {...register('config.video_recording')}
                                                                        className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-600"
                                                                    />
                                                                </div>
                                                                <div className="ml-3">
                                                                    <label htmlFor="video-recording" className="text-sm font-medium leading-6 text-gray-900">
                                                                        Live Video Recording
                                                                    </label>
                                                                    <p className="text-sm text-gray-500">
                                                                        Record video during training (slower).
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}

                                                    {/* Steps and Resources */}
                                                    <div className="sm:col-span-3">
                                                        <label className="block text-sm font-medium leading-6 text-gray-900">
                                                            Total Timesteps
                                                        </label>
                                                        <div className="mt-2">
                                                            <input
                                                                type="number"
                                                                {...register('config.total_timesteps', { valueAsNumber: true })}
                                                                className="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                            />
                                                        </div>
                                                    </div>

                                                    <div className="sm:col-span-3">
                                                        <label className="block text-sm font-medium leading-6 text-gray-900">
                                                            Learning Rate (Default: 0.00025)
                                                        </label>
                                                        <div className="mt-2">
                                                            <input
                                                                type="number"
                                                                step="0.00001"
                                                                {...register('config.learning_rate', { valueAsNumber: true })}
                                                                className="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                            />
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="border-t border-gray-200 pt-6">
                                                    <h4 className="text-sm font-medium text-gray-900">Resource Limits</h4>
                                                    <div className="mt-4 grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-6">
                                                        <div className="sm:col-span-2">
                                                            <label className="block text-sm font-medium leading-6 text-gray-900">
                                                                CPU Cores
                                                            </label>
                                                            <div className="mt-2">
                                                                <input
                                                                    type="number"
                                                                    step="0.5"
                                                                    {...register('config.resource_limits.cpu_cores', { valueAsNumber: true })}
                                                                    className="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                                />
                                                            </div>
                                                        </div>
                                                        <div className="sm:col-span-2">
                                                            <label className="block text-sm font-medium leading-6 text-gray-900">
                                                                Memory (GB)
                                                            </label>
                                                            <div className="mt-2">
                                                                <input
                                                                    type="number"
                                                                    step="0.5"
                                                                    {...register('config.resource_limits.memory_gb', { valueAsNumber: true })}
                                                                    className="block w-full rounded-md border-0 py-1.5 text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
                                                                />
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="mt-5 sm:mt-6 sm:grid sm:grid-flow-row-dense sm:grid-cols-2 sm:gap-3">
                                                    <button
                                                        type="submit"
                                                        disabled={isLoading}
                                                        className="inline-flex w-full justify-center rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 sm:col-start-2 disabled:opacity-50"
                                                    >
                                                        {isLoading ? 'Creating...' : 'Create Container'}
                                                    </button>
                                                    <button
                                                        type="button"
                                                        className="mt-3 inline-flex w-full justify-center rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 sm:col-start-1 sm:mt-0"
                                                        onClick={onClose}
                                                    >
                                                        Cancel
                                                    </button>
                                                </div>
                                            </form>
                                        </div>
                                    </div>
                                </div>
                            </Dialog.Panel>
                        </Transition.Child>
                    </div>
                </div>
            </Dialog>
        </Transition.Root>
    );
}
