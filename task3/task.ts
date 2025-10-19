export interface TaskExecution {
  id?: string;
  startedAt?: string;
  finishedAt?: string;
  output?: string;
  status?: 'PENDING' | 'RUNNING' | 'SUCCESS' | 'FAILED';
}

export interface Task {
  id?: string;
  name: string;
  command: string;
  schedule?: string;
  lastRunAt?: string;
  lastStatus?: 'SUCCESS' | 'FAILED' | 'PENDING' | 'RUNNING';
  executions?: TaskExecution[];
}
