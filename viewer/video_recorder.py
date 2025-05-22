import imageio
import os
import queue
import threading

class VideoRecorder:
    def __init__(self, root_dir=None, fps=20, max_concurrent_saves=3):
        """VideoRecorder with multi-threaded saving but limits concurrent saves."""
        self.save_dir = root_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.fps = fps
        self.frames = []

        # Thread management
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        # Semaphore to limit concurrent saves
        self.max_concurrent_saves = max_concurrent_saves
        self.semaphore = threading.Semaphore(max_concurrent_saves)

    def _worker(self):
        """Processes video saving jobs while respecting the limit."""
        while True:
            file_name, frames = self.queue.get()
            if file_name is None:
                break  # Exit the worker thread

            # Limit concurrent saves
            self.semaphore.acquire()
            try:
                self._save_video(file_name, frames)
            finally:
                self.semaphore.release()  # Release after saving

            self.queue.task_done()

    def _save_video(self, file_name, frames):
        """Saves the video file."""
        if not frames:
            return  # Skip if no frames
        path = os.path.join(self.save_dir, file_name)
        imageio.mimsave(path, frames, fps=self.fps)

    def update_save_dir(self, root_dir):
        if root_dir is not None:
            self.save_dir = root_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

    def append(self, frame):
        """Adds a frame to the buffer."""
        self.frames.append(frame)

    def reset(self):
        """Clears stored frames."""
        self.frames = []

    def save(self, file_name):
        """Queues the video saving job while respecting concurrency limits."""
        if not self.frames:
            return  # Skip saving if no frames

        self.queue.put((file_name, self.frames[:]))  # Copy frames
        self.reset()  # Free memory after queuing

    def stop(self):
        """Stops the worker thread safely."""
        self.queue.put((None, None))  # Signal thread to exit
        self.worker_thread.join()