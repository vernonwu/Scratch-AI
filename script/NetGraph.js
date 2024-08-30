class NetGraph {
    constructor(graphName) {
        this.nodeList = [];
        this.edgeList = [];
        this.parameterList = {};
        this.graphName = graphName;
        this.inputDim = null;
        this.outputDim = null;
        this.initializeParameterList();
    }

    initializeParameterList() {
        switch (this.graphName) {
            case 'input':
                this.parameterList["input dimension"] = [3,48,48];
                break;
            case 'output':
                this.parameterList["output dimension"] = 10;
                break;
            case 'Conv2d':
                this.parameterList["Activate Function"] = 'Relu';
                this.parameterList.padding = 0;
                this.parameterList.kernel_size = 3;
                this.parameterList.stride = 1;
                break;
            case 'Linear':
                this.parameterList["Activate Function"] = 'Relu';
                this.parameterList["in_features"] = 48;
                this.parameterList["out_features"] = 10;
                break;
            case 'pooling':
                this.parameterList["Pooling type"] = 'Max';
                this.parameterList.kernel_size = 3;
                this.parameterList.stride = 1;
                break;
            case 'concatenate':
                this.parameterList["dim"] = 1;
                break;
            default:
                console.warn(`Unsupported graph name: ${this.graphName}`);
        }
    }

    calculateOutputDimensions(inputDim) {
        switch (this.graphName) {
            case 'input' || 'output':
                return inputDim;
            case 'concatenate':
                //TODO: modify this
                return inputDim;
            case 'Conv2d':
                return calculateConv2dOutput(inputDim);
            case 'pooling':
                return calculatePoolingOutput(inputDim);
            case 'Linear':
                return calculateLinearOutput(inputDim);
            default:
                console.warn(`Unsupported module name: ${moduleName}`);
                return inputDim;
        }
    }

    //TODO: implement for multiple batches

    calculateConv2dOutput(inputDim) {
        const [inChannels, inHeight, inWidth] = inputDim;
        const kernel_size = this.parameterList.kernel_size;
        const padding = this.parameterList.padding;
        const stride = this.parameterList.stride;
        const outHeight = Math.floor((inHeight + 2 * padding - kernel_size) / stride + 1);
        const outWidth = Math.floor((inWidth + 2 * padding - kernel_size) / stride + 1);
        return `${inChannels},${outHeight},${outWidth}`;
    }

    calculatePoolingOutput(inputDim) {
        const [inChannels, inHeight, inWidth] = inputDim;
        const kernel_size = this.parameterList.kernel_size;
        const stride = this.parameterList.stride;
        const outHeight = Math.floor((inHeight - kernel_size) / stride + 1);
        const outWidth = Math.floor((inWidth - kernel_size) / stride + 1);
        return `${inChannels},${outHeight},${outWidth}`;
    }

    calculateLinearOutput(inputDim) {
        const outFeatures = this.parameterList["out features"];
        return `${outFeatures}`;
    }

    addNode(Node) {
        if (!this.nodeList.includes(Node)) {
            this.nodeList.push(Node);
            let index = this.nodeList.indexOf(Node);
            this.edgeList = this.augmentMatrix(this.edgeList, index);
            return true;
        } else {
            return false;
        }
    }

    addEdge(start, end) {
        if (!this.nodeList.includes(start) || !this.nodeList.includes(end)) {
            return false;
        }
        let startIndex = this.nodeList.indexOf(start);
        let endIndex = this.nodeList.indexOf(end);
        this.edgeList[startIndex][endIndex] = 1;
    }

    deleteNode(Node) {
        if (!this.nodeList.includes(Node)) {
            return false;
        } else {
            let index = this.nodeList.indexOf(Node);
            this.nodeList.splice(index, 1);
            this.edgeList = this.augmentMatrix2(this.edgeList, index);
            return true;
        }
    }

    deleteEdge(start, end) {
        if (!this.nodeList.includes(start) || !this.nodeList.includes(end)) {
            return false;
        }
        let startIndex = this.nodeList.indexOf(start);
        let endIndex = this.nodeList.indexOf(end);
        this.edgeList[startIndex][endIndex] = 0;
    }

    setInputDim(inputDim) {
        this.inputDim = inputDim;
    }

    setOutputDim(outputDim) {
        this.outputDim = outputDim;
    }

    generateZeroMatrix(N) {
        let matrix = [];
        for (let i = 0; i < N; i++) {
            matrix[i] = [];
            for (let j = 0; j < N; j++) {
                matrix[i][j] = 0;
            }
        }
        return matrix;
    }

    augmentMatrix(matrix, pos) {
        const len = matrix.length;
        for (let row = 0; row < len; row++) {
            matrix[row].splice(pos, 0, 0);
        }
        let newRow = Array(len + 1).fill(0);
        matrix.splice(pos, 0, newRow);
        return matrix;
    }

    augmentMatrix2(matrix, pos) {
        const len = matrix.length;
        for (let row = 0; row < len; row++) {
            matrix[row].splice(pos, 1);
        }
        matrix.splice(pos, 1);
        return matrix;
    }

    checkMatrix(matrix) {
        let zeroRowIndex = -1;
        let zeroColIndex = -1;
        let rowCount = 0;
        let colCount = 0;

        for (let i = 0; i < matrix.length; i++) {
            let rowSum = 0;
            let colSum = 0;
            for (let j = 0; j < matrix[i].length; j++) {
                rowSum += matrix[i][j];
                colSum += matrix[j][i];
            }
            if (rowSum === 0) {
                zeroRowIndex = i;
                rowCount++;
            }
            if (colSum === 0) {
                zeroColIndex = i;
                colCount++;
            }
        }

        if (rowCount === 1 && colCount === 1) {
            return [zeroColIndex, zeroRowIndex];
        } else {
            return false;
        }
    }

    topologicalSort() {
        let indegree = new Array(this.nodeList.length).fill(0);
        let sortedNodes = [];
        let queue = [];

        // Calculate indegree for each node
        for (let i = 0; i < this.edgeList.length; i++) {
            for (let j = 0; j < this.edgeList[i].length; j++) {
                if (this.edgeList[i][j] === 1) {
                    indegree[j]++;
                }
            }
        }

        // Add all nodes with indegree 0 to the queue
        for (let i = 0; i < indegree.length; i++) {
            if (indegree[i] === 0) {
                queue.push(i);
            }
        }

        // Process nodes in queue
        while (queue.length > 0) {
            let node = queue.shift();
            sortedNodes.push(node);

            for (let i = 0; i < this.edgeList[node].length; i++) {
                if (this.edgeList[node][i] === 1) {
                    indegree[i]--;
                    if (indegree[i] === 0) {
                        queue.push(i);
                    }
                }
            }
        }

        // If all nodes are not sorted, the graph has a cycle
        if (sortedNodes.length !== this.nodeList.length) {
            console.error("Graph has a cycle!");
            return false;
        }

        // Reorder nodeList and edgeList based on sortedNodes
        let newNodeList = sortedNodes.map(index => this.nodeList[index]);
        let newEdgeList = this.generateZeroMatrix(this.nodeList.length);

        for (let i = 0; i < sortedNodes.length; i++) {
            for (let j = 0; j < sortedNodes.length; j++) {
                newEdgeList[i][j] = this.edgeList[sortedNodes[i]][sortedNodes[j]];
            }
        }

        this.nodeList = newNodeList;
        this.edgeList = newEdgeList;
        return true;
    }

    printGraph_bfsHelp(matrix, namelist, startpos, endpos) {
        let paths = [];
        let queue = [[startpos]];
        let codeArea = document.getElementById("code-area");

        while (queue.length > 0) {
            let path = queue.shift();
            let node = path[path.length - 1];

            if (node === endpos) {
                paths.push(path);
            }

            for (let i = 0; i < matrix[node].length; i++) {
                if (matrix[node][i] === 1 && !path.includes(i)) {
                    let new_path = path.slice();
                    new_path.push(i);
                    queue.push(new_path);
                }
            }
        }

        codeArea.innerHTML = '';

        for (let path of paths) {
            let pathNames = path.map(node => namelist[node]);
            codeArea.innerHTML += pathNames.join('->') + "\n\n";
            console.log(pathNames.join('->'));
        }

        return paths;
    }

    printGraph_bfs() {
        let codeArea = document.getElementById("code-area");

        if (!this.topologicalSort()) {
            codeArea.innerHTML = "Error: The graph contains a cycle!";
            return;
        }

        if (this.nodeList[0].graphName !== "input" || this.nodeList[this.nodeList.length - 1].graphName !== "output") {
            codeArea.innerHTML = "Error: The first node must be 'input' and the last node must be 'output'.";
            return;
        }

        let GraphStartNode, GraphEndNode;
        [GraphStartNode, GraphEndNode] = this.checkMatrix(this.edgeList);

        // set input dimension if not set
        let inputNode = this.nodeList[0];
        let inputDim = inputNode.parameterList["input dimension"];
        if (!inputDim) {
            inputDim = [3, 48, 48];
            inputNode.parameterList["input dimension"] = inputDim;
        }
        inputNode.setInputDim(inputDim);
        inputNode.setOutputDim(inputDim);

        //TODO: Autodimension logic

        let namelist = [];
        this.nodeList.forEach(node => {
            namelist.push(node.graphName + ":" + JSON.stringify(node.parameterList));
        });

        let paths = this.printGraph_bfsHelp(this.edgeList, namelist, GraphStartNode, GraphEndNode);

        // record all parent nodes for each node
        let parentNodes = {};

        for (let path of paths) {
            for (let i = 1; i < path.length; i++) {
                let node = path[i];
                let parent = path[i - 1];
                if (!parentNodes[node]) {
                    parentNodes[node] = [];
                }
                if (!parentNodes[node].includes(parent)) {
                    parentNodes[node].push(parent);
                }
            }
        }

        // if a node has more than one parent, check if it is a concatenation layer
        for (let node in parentNodes) {
            if (parentNodes[node].length > 1) {
                if (this.nodeList[node].graphName !== 'concatenate') {
                    codeArea.innerHTML = `Error: Concatenation layer required before node ${this.nodeList[node].graphName}.`;
                    return;
                }
            }
        }

        let codeGenerator = new PyTorchCodeGenerator(this, parentNodes);
        let generatedCode = codeGenerator.generate();
        codeArea.innerHTML += "--------------------------------\n";
        codeArea.innerHTML += generatedCode;

        return generatedCode;

    }
}

