import { Queue } from 'bullmq';
import { redis } from './lib.js';
export const imageQueue = new Queue('passport-images', { connection: redis });
export async function enqueue(fileId:string){ return imageQueue.add('process', {fileId}, { jobId:fileId, attempts:3, backoff:{type:'exponential',delay:1500}, removeOnComplete:5000, removeOnFail:false }); }
