export class QueryableWorker {
    constructor(url, defaultListener, onError) {
        this.worker = new Worker(url);
        this.listeners = {};
        this.defaultListener = defaultListener ?? (() => { });

        if (onError) {
            this.worker.onerror = onError;
        }

        this.worker.onmessage = (event) => {
            if (event.data instanceof Object &&
                Object.hasOwn(event.data, "queryMethodListener") &&
                Object.hasOwn(event.data, "queryMethodArguments") &&
                this.listeners[event.data.queryMethodListener]
            ) {
                this.listeners[event.data.queryMethodListener].apply(
                    this,
                    event.data.queryMethodArguments
                );
            } else {
                this.defaultListener.call(this, event.data);
            }
        };
    }


    postMessage = (message) => {
        this.worker.postMessage(message);
    };

    terminate = () => {
        this.worker.terminate();
    };

    addListener = (name, listener) => {
        this.listeners[name] = listener;
    };

    removeListener = (name) => {
        delete this.listeners[name];
    };

    sendQuery = (queryMethod, ...queryMethodArguments) => {
        if (!queryMethod) {
            throw new TypeError(
                "QueryableWorker.sendQuery takes at least one argument"
            );
        }
        this.worker.postMessage({
            queryMethod,
            queryMethodArguments,
        });
    };

}