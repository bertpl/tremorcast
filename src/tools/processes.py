import os
import signal

import psutil


def kill_all_child_processes(timeout: float = 5.0, sigterm: bool = True, sigkill: bool = True, verbose: bool = False):

    # --- handle verbosity --------------------------------
    if verbose:
        log = print
    else:
        log = lambda *a, **k: None

    # --- fetch processes ---------------------------------
    main_process = psutil.Process(os.getpid())
    children = main_process.children(recursive=True)

    log(f"Attempting to kill {len(children):_} child processes.")
    log(f"Main process : {main_process.pid}")
    log(f"Children     : {', '.join([str(p.pid) for p in children])}")

    # --- send sigterm / sigkill to children --------------

    # determine which signals to send
    signals = []
    if sigterm:
        signals.append(signal.SIGTERM)
    if sigkill:
        signals.append(signal.SIGKILL)

    # main loop
    alive = children
    for sig in signals:

        if len(alive) > 0:

            log(f"Sending {sig.name} to {len(alive):_} child processes...".ljust(60), end="")

            for p in alive:
                try:
                    p.send_signal(sig)
                except psutil.NoSuchProcess:
                    pass

            gone, alive = psutil.wait_procs(children, timeout=timeout)

            log(f"{len(gone):_} killed, {len(alive):_} still alive.")
