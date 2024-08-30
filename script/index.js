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
        codeArea.innerHTML += generatedCode;
    }
}

class PyTorchCodeGenerator {
    constructor(graph, parentNodes) {
        this.graph = graph;
        this.parentNodes = parentNodes;
        this.layerNames = {};
        this.generateModuleName();
    }

    generateModuleName(){
        let moduleCount = {};
        for (let i = 0; i < this.graph.nodeList.length; i++) {
            let node = this.graph.nodeList[i];
            let layerName = node.graphName;
            if (!moduleCount[layerName]) {
                moduleCount[layerName] = 1;
            } else {
                moduleCount[layerName]++;
            }
            this.layerNames[i] = layerName + moduleCount[layerName];
        }
    }

    generatePythonLine(index){
        let layerName = this.layerNames[index];
        let parent = this.parentNodes[index];
        let node = this.graph.nodeList[index];
        let parentNames = parent.map(p => this.layerNames[p]);
        if (parent.length > 1) {
            return `        ${layerName}_output = torch.cat([${parentNames.map(p => p + '_output').join(', ')}], dim=${node.parameterList.dim})\n`;
        }
        else {
            return `        ${layerName}_output = self.${layerName}(${parentNames[0] + '_output'})\n`;
        }
    }

    generate() {
        let code = "import torch\nimport torch.nn as nn\n\n";
        code += "class MyModel(nn.Module):\n";
        code += "    def __init__(self):\n";
        code += "        super(MyModel, self).__init__()\n";

        // Define layers
        for (let i = 1; i < this.graph.nodeList.length-1; i++) {
            let node = this.graph.nodeList[i];
            let layerType = node.graphName;
            let layerName = this.layerNames[i];
            let layerParams = node.parameterList;
            let layerParamsStr = JSON.stringify(layerParams);
            code += `        self.${layerName} = nn.${layerType}(${layerParamsStr})\n`;
        }

        // Define forward pass
        code += "\n    def forward(self, x):\n";
        code += "        input0_output = x \n";

        for (let i = 1; i < this.graph.nodeList.length; i++) {
            let node = this.graph.nodeList[i];
            code += this.generatePythonLine(i);
        }

        code += "        return output" + (this.graph.nodeList.length - 1) + "_output\n";
        return code;
    }

}

var globalclicktime = 0;
let arrowId = 0;
let componentWidth = 120;
let componentHeight = 180;
var tool = 0;
let myGraph = new NetGraph('Main-Graph');
let totalNodeList = [];
let paraExample =
    { 'kernel_size': 3, "padding": 0, "stride": 1, "input dimension": "3,48,48", "output dimension": "10",
        "in_features": "1024", "out_features": "512", "Activate Function": "Relu", "Pooling type": "Max", "dim": "1"
    };
const ActivateFunctionList = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', 'Softmax', 'Swish'];
const PoolingTypeList = ['Max', "Min", "Average"];

let canvas = document.getElementById("canvas");
let startNode, endNode;

// const offcanvasElementList = document.querySelectorAll('.offcanvas')
// const offcanvasList = [...offcanvasElementList].map(offcanvasEl => new bootstrap.Offcanvas(offcanvasEl))


function sign(x) {
    return (x > 0) - (x < 0) || +x;
}
canvas.addEventListener('mousemove', function (event) {
    if (globalclicktime == 1) {
        canvas = document.getElementById("canvas");
        var offsetX = event.clientX - canvas.getBoundingClientRect().left;
        var offsetY = event.clientY - canvas.getBoundingClientRect().top;

        line = document.getElementById(arrowId);
        offsetX = parseFloat(offsetX);
        offsetY = parseFloat(offsetY);
        var pathData = line.getAttribute("d");
        var coordinates = pathData.match(/[-+]?\d*\.?\d+/g);
        coordinates = coordinates.map(coord => parseFloat(coord));
        coordinates[2] = coordinates[0] + (offsetX - coordinates[0]) / 3;
        coordinates[3] = coordinates[1] + (offsetY - coordinates[1]) / 3;
        coordinates[4] = coordinates[0] + 2 * (offsetX - coordinates[0]) / 3;
        coordinates[5] = coordinates[1] + 2 * (offsetY - coordinates[1]) / 3;
        coordinates[6] = offsetX;
        coordinates[7] = offsetY;
        line.setAttribute("d", "M " + coordinates[0] + " " + coordinates[1] + " C " +
            coordinates[2] + " " + coordinates[3] + ", " +
            coordinates[4] + " " + coordinates[5] + ", " +
            coordinates[6] + " " + coordinates[7]);
    }
});

let input = document.querySelector('input[name="searchInfo"]');
input.addEventListener('input', function () {
    const searchText = input.value.toLowerCase();

    let draggables = document.querySelectorAll('.draggable');
    draggables.forEach(function (image) {
        const imgName = image.getAttribute('imgname').toLowerCase();
        if (imgName.includes(searchText)) {
            image.style.display = 'block';
        } else {
            image.style.display = 'none';
        }
    });
});

const card = document.getElementById('infocard');
cardimg = document.getElementById('cardimg');
let draggables = document.querySelectorAll(".draggable");
draggables.forEach(function (image) {

    image.addEventListener('mouseover', function (event) {
        card.style.visibility = 'visible';
        cardimg.src = image.getAttribute('src');
        if(image.getAttribute('src')=="image/concatenate.png"){
            cardimg.style.height=componentHeight - 80+'px';
        }
        else{
            cardimg.style.height=componentHeight-30+'px';
        }
        let cardtitle = document.getElementById('cardtitle')
        cardtitle.textContent = image.getAttribute('imgname');
        let cardtext = document.getElementById('cardtext')
        cardtext.textContent = image.getAttribute('cardtext');
    });

    image.addEventListener('mouseout', function (event) {
        card.style.visibility = 'hidden';
    });

    image.addEventListener('mousedown', function (event) {
        card.style.visibility = 'hidden';
    });
});

function selectOnclick(event) {
    tool = 0;
    document.getElementById("selecttool").style.backgroundColor = "#e6e6e6";
    document.getElementById("deletetool").style.backgroundColor = "#f2f2f2";
    document.getElementById("paratool").style.backgroundColor = "#f2f2f2";
    buttonClickHandler();
    console.log(myGraph);
}

function deleteOnclick(event) {
    tool = 1;
    document.getElementById("selecttool").style.backgroundColor = "#f2f2f2";
    document.getElementById("deletetool").style.backgroundColor = "#e6e6e6";
    document.getElementById("paratool").style.backgroundColor = "#f2f2f2";
    buttonClickHandler();

}

function ParaToolOnclick(event) {
    tool = 2;
    document.getElementById("selecttool").style.backgroundColor = "#f2f2f2";
    document.getElementById("deletetool").style.backgroundColor = "#f2f2f2";
    document.getElementById("paratool").style.backgroundColor = "#e6e6e6";
}

function GenerateToolOnclick(event) {
    myGraph.printGraph_bfs();
}

function allowDrop(event) {
    event.preventDefault();
    console.log("dragzone");
}

function dotclick_handler(event) {
    if (tool == 0) {
        if (globalclicktime == 0) {
            globalclicktime = 1;
            startNode = totalNodeList[event.target.parentNode.getAttribute("Nodepos")];
            svg = document.getElementById('main-svg');
            var offsetX = (event.target.getBoundingClientRect().left - svg.getBoundingClientRect().left);
            var offsetY = (event.target.getBoundingClientRect().top - svg.getBoundingClientRect().top);
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            line.setAttribute("d", "M " + offsetX + " " + offsetY + " C " + offsetX + " " + offsetY + ", " + offsetX + " " + offsetY + ", " + offsetX + " " + offsetY);
            line.setAttribute("stroke", "black"); // 设置描边颜色
            line.setAttribute("fill", "transparent");
            line.setAttribute('marker-end', 'url(#arrow)');
            line.setAttribute('marker-start', 'url(#circle)')
            line.setAttribute("stroke-width", 1);
            line.id = arrowId;
            // line.setAttribute("shiftY", Math.floor(-event.target.parentNode.getBoundingClientRect().height / 2 - event.target.parentNode.getBoundingClientRect().top + event.target.getBoundingClientRect().top) + 4);
            line.style.zIndex = 2;
            line.onclick = function (event) {
                if (tool == 1) {
                    deleteArrow(event.target.id);
                }
            }
            svg.appendChild(line);
            var arrowList = event.target.getAttribute('arrow-list');
            if (!arrowList) {
                var initialList = [];
                initialList.push(arrowId);
                initialList.push("start");
                event.target.setAttribute('arrow-list', JSON.stringify(initialList));

            }
            else {
                arrowList = JSON.parse(arrowList);
                arrowList.push(arrowId);
                arrowList.push("start");
                event.target.setAttribute('arrow-list', JSON.stringify(arrowList));
            }

        }


        else if (globalclicktime == 1) {
            globalclicktime = 0;
            var arrowList = event.target.getAttribute('arrow-list');
            endNode = totalNodeList[event.target.parentNode.getAttribute("Nodepos")];
            if (!arrowList) {
                var initialList = [];
                initialList.push(arrowId);
                initialList.push("end");
                event.target.setAttribute('arrow-list', JSON.stringify(initialList));

            }
            else {
                arrowList = JSON.parse(arrowList);
                arrowList.push(arrowId);
                arrowList.push("end");
                event.target.setAttribute('arrow-list', JSON.stringify(arrowList));
            }
            var offsetX = (event.target.getBoundingClientRect().left - svg.getBoundingClientRect().left);
            var offsetY = (event.target.getBoundingClientRect().top - svg.getBoundingClientRect().top);
            line = document.getElementById(arrowId);
            var pathData = line.getAttribute("d");
            var coordinates = pathData.match(/[-+]?\d*\.?\d+/g);
            coordinates = coordinates.map(coord => parseFloat(coord));
            coordinates[2] = coordinates[0] + (offsetX - coordinates[0]) / 3;
            coordinates[3] = coordinates[1] + (offsetY - coordinates[1]) / 3;
            coordinates[4] = coordinates[0] + 2 * (offsetX - coordinates[0]) / 3;
            coordinates[5] = coordinates[1] + 2 * (offsetY - coordinates[1]) / 3;
            coordinates[6] = offsetX;
            coordinates[7] = offsetY;
            line.setAttribute("d", "M " + coordinates[0] + " " + coordinates[1] + " C " +
                coordinates[2] + " " + coordinates[3] + ", " +
                coordinates[4] + " " + coordinates[5] + ", " +
                coordinates[6] + " " + coordinates[7]);
            line.classList.add("line");
            myGraph.addEdge(startNode, endNode);
            arrowId++;
            adjustAllArrow();
        }
    }
}

function updateArrow(element) {
    for (i = 1; i < 5; i++) {
        var arrowList = element.children[i].getAttribute('arrow-list');
        const svg = document.getElementById('main-svg');
        var offsetX = (element.children[i].getBoundingClientRect().left - svg.getBoundingClientRect().left);
        var offsetY = (element.children[i].getBoundingClientRect().top - svg.getBoundingClientRect().top);
        if (!arrowList) {
            continue;
        }
        else {
            arrowList = JSON.parse(arrowList);
            for (j = 0; j < arrowList.length / 2; j++) {
                if (arrowList[2 * j + 1] == "end") {
                    line = document.getElementById(arrowList[2 * j])
                    var pathData = line.getAttribute("d");
                    var coordinates = pathData.match(/[-+]?\d*\.?\d+/g);
                    coordinates = coordinates.map(coord => parseFloat(coord));
                    coordinates[2] = coordinates[0] + (offsetX - coordinates[0]) / 3;
                    coordinates[3] = coordinates[1] + (offsetY - coordinates[1]) / 3;
                    coordinates[4] = coordinates[0] + 2 * (offsetX - coordinates[0]) / 3;
                    coordinates[5] = coordinates[1] + 2 * (offsetY - coordinates[1]) / 3;
                    coordinates[6] = offsetX;
                    coordinates[7] = offsetY;
                    line.setAttribute("d", "M " + coordinates[0] + " " + coordinates[1] + " C " +
                        coordinates[2] + " " + coordinates[3] + ", " +
                        coordinates[4] + " " + coordinates[5] + ", " +
                        coordinates[6] + " " + coordinates[7]);
                }
                else if (arrowList[2 * j + 1] == "start") {
                    line = document.getElementById(arrowList[2 * j])
                    var pathData = line.getAttribute("d");
                    var coordinates = pathData.match(/[-+]?\d*\.?\d+/g);
                    coordinates = coordinates.map(coord => parseFloat(coord));
                    coordinates[0] = offsetX;
                    coordinates[1] = offsetY;
                    coordinates[2] = coordinates[6] - 2 * (offsetX - coordinates[0]) / 3;
                    coordinates[3] = coordinates[7] - 2 * (offsetY - coordinates[1]) / 3;
                    coordinates[4] = coordinates[6] - (offsetX - coordinates[0]) / 3;
                    coordinates[5] = coordinates[7] - (offsetY - coordinates[1]) / 3;
                    line.setAttribute("d", "M " + coordinates[0] + " " + coordinates[1] + " C " +
                        coordinates[2] + " " + coordinates[3] + ", " +
                        coordinates[4] + " " + coordinates[5] + ", " +
                        coordinates[6] + " " + coordinates[7]);
                }
            }
        }
    }
    adjustAllArrow();
}

function adjustAllArrow() {
    let lines = document.querySelectorAll('.line');
    lines.forEach(line => {
        let adjustId = line.id;
        let startDirection, endDirection;
        document.querySelectorAll(".dot").forEach(dot => {
            var arrowList = dot.getAttribute('arrow-list');
            if (!arrowList) {
                return;
            }
            arrowList = JSON.parse(arrowList);
            for (j = 0; j < arrowList.length / 2; j++) {
                if (arrowList[2 * j] == adjustId && arrowList[2 * j + 1] == "end") {
                    endDirection = Array.from(dot.parentNode.childNodes).indexOf(dot);
                }
                else if (arrowList[2 * j] == adjustId && arrowList[2 * j + 1] == "start") {
                    startDirection = Array.from(dot.parentNode.childNodes).indexOf(dot);
                }
            }
        })
        var pathData = line.getAttribute("d");
        var coordinates = pathData.match(/[-+]?\d*\.?\d+/g);
        coordinates = coordinates.map(coord => parseFloat(coord));
        coordinates[2] = coordinates[6] - 2 * (coordinates[6] - coordinates[0]) / 3;
        coordinates[3] = coordinates[7] - 2 * (coordinates[7] - coordinates[1]) / 3;
        coordinates[4] = coordinates[6] - (coordinates[6] - coordinates[0]) / 3;
        coordinates[5] = coordinates[7] - (coordinates[7] - coordinates[1]) / 3;
        if (startDirection == 1 || startDirection == 2) {
            coordinates[3] = coordinates[1] + sign(startDirection - 1.5) * Math.abs(coordinates[7]-coordinates[1]);
        }
        else if (startDirection == 3 || startDirection == 4) {
            coordinates[2] = coordinates[0] + sign(startDirection - 3.5) * Math.abs(coordinates[6]-coordinates[0]);
        }

        if (endDirection == 1 || endDirection == 2) {
            coordinates[5] = coordinates[7] + sign(endDirection - 1.5) * Math.abs(coordinates[7]-coordinates[1]);
        }
        else if (endDirection == 3 || endDirection == 4) {
            coordinates[4] = coordinates[6] + sign(endDirection - 3.5) * Math.abs(coordinates[6]-coordinates[0]);
        }

        line.setAttribute("d", "M " + coordinates[0] + " " + coordinates[1] + " C " +
            coordinates[2] + " " + coordinates[3] + ", " +
            coordinates[4] + " " + coordinates[5] + ", " +
            coordinates[6] + " " + coordinates[7]);
    })
}

function deleteArrow(id) {
    var dots = document.querySelectorAll('.dot');
    dots.forEach(dot => {
        ArrowListTemp = dot.getAttribute('arrow-list');
        if (!ArrowListTemp) {
        }
        else {
            ArrowListTemp = JSON.parse(ArrowListTemp);
            for (num = 0; num < ArrowListTemp.length / 2; num++) {
                if (ArrowListTemp[2 * num] == id) {
                    if (ArrowListTemp[2 * num + 1] == 'end') {
                        endNode = totalNodeList[dot.parentNode.getAttribute("Nodepos")];
                    }
                    else if (ArrowListTemp[2 * num + 1] == 'start') {
                        startNode = totalNodeList[dot.parentNode.getAttribute("Nodepos")];
                    }
                    ArrowListTemp.splice(2 * num, 2);
                    dot.setAttribute('arrow-list', JSON.stringify(ArrowListTemp));
                    break;
                }
            }
        }

    });
    myGraph.deleteEdge(startNode, endNode);
    document.getElementById(id).remove();
}

function drop(event) {
    event.preventDefault();
    var data = event.dataTransfer.getData("text");
    var offX = event.dataTransfer.getData("offX");
    var offY = event.dataTransfer.getData("offY");
    var id = event.dataTransfer.getData('id');
    var GraphName = event.dataTransfer.getData('GraphName');
    if (id == '0') {
        var element = document.createElement("div");
        let ran = Math.floor(Math.random() * 1000) + 10000;
        var image = document.createElement('img');
        image.src = data;
        image.style.position = 'relative'
        image.style.top = "2%";
        image.style.left = '2%';
        image.style.height = '96%';
        image.style.width = '96%';
        element.draggable = "true";
        element.classList.add("component");
        element.setAttribute('GraphName', GraphName);
        element.style.width = componentWidth + 'px';
        element.style.height = componentHeight + 'px';
        if(GraphName == "concatenate"){
            element.style.width = componentWidth + 'px';
            element.style.height = componentHeight - 60 + 'px';
        }
        element.style.position = "absolute";
        element.style.zIndex = 2;
        element.ondragstart = function (event) {
            event.dataTransfer.setData('offX', offX);
            event.dataTransfer.setData('offY', offY);
            if (event.target.parentNode.id != "canvas") {
                event.dataTransfer.setData('id', event.target.parentNode.id)
            }
            else {
                event.dataTransfer.setData('id', event.target.id)
            }
        };
        element.onclick = function (event) {
            if (tool == 1) {
                var main_div = event.target.parentNode;
                for (i = 1; i < 5; i++) {
                    var arrowList = main_div.children[i].getAttribute('arrow-list');
                    if (!arrowList) {
                        continue;
                    }
                    else {
                        arrowList = JSON.parse(arrowList);

                        for (j = 0; j < arrowList.length / 2;) {
                            deleteArrow(arrowList[2 * j]);
                            j++;
                        }
                    }
                }
                myGraph.deleteNode(totalNodeList[event.target.parentNode.getAttribute("Nodepos")]);
                main_div.remove();
            }
            else if (tool == 2) {
                let targetNode = totalNodeList[event.target.parentNode.getAttribute("Nodepos")];
                let paracard = document.getElementById('paracard');
                let paracardHeader = document.getElementById('paracardHeader');
                let paracardList = document.getElementById('paracardList');
                paracardHeader.childNodes[1].innerHTML = targetNode.graphName;
                paracard.style.right = "0%";

                while (paracardList.firstChild) {
                    paracardList.removeChild(paracardList.firstChild);
                }

                let keys = Object.keys(targetNode.parameterList);
                keys.forEach(key => {
                    if (key == "Activate Function") {
                        var pararow = document.createElement('li');
                        var parainput = document.createElement('select');
                        parainput.setAttribute("Nodepos", totalNodeList.indexOf(targetNode));
                        parainput.setAttribute("Parameter", key);
                        parainput.addEventListener('change', function () {
                            let targetNode = totalNodeList[this.getAttribute("Nodepos")];
                            let parameter = this.getAttribute("Parameter");
                            targetNode.parameterList[parameter] = this.value;
                        });
                        ActivateFunctionList.forEach(ActivateFunction => {
                            var option = document.createElement('option');
                            option.value = ActivateFunction;
                            option.innerHTML = ActivateFunction;
                            parainput.appendChild(option);
                        })
                        pararow.classList.add('list-group-item');
                        if (targetNode.parameterList[key]) {
                            parainput.value = targetNode.parameterList[key];

                        }

                        pararow.innerHTML = key + ':';
                        pararow.appendChild(parainput);
                        paracardList.appendChild(pararow);
                        return;
                    }
                    else if (key == "Pooling type") {
                        var pararow = document.createElement('li');
                        var parainput = document.createElement('select');
                        parainput.setAttribute("Nodepos", totalNodeList.indexOf(targetNode));
                        parainput.setAttribute("Parameter", key);
                        parainput.addEventListener('change', function () {
                            let targetNode = totalNodeList[this.getAttribute("Nodepos")];
                            let parameter = this.getAttribute("Parameter");
                            targetNode.parameterList[parameter] = this.value;
                        });
                        PoolingTypeList.forEach(PoolingType => {
                            var option = document.createElement('option');
                            option.value = PoolingType;
                            option.innerHTML = PoolingType;
                            parainput.appendChild(option);
                        })
                        pararow.classList.add('list-group-item');
                        if (targetNode.parameterList[key]) {
                            parainput.value = targetNode.parameterList[key];

                        }

                        pararow.innerHTML = key + ':';
                        pararow.appendChild(parainput);
                        paracardList.appendChild(pararow);
                        return;
                    }

                    var pararow = document.createElement('li');
                    var parainput = document.createElement('input');
                    parainput.style.width = "100%";
                    parainput.setAttribute("Nodepos", totalNodeList.indexOf(targetNode));
                    parainput.setAttribute("Parameter", key)
                    parainput.addEventListener('input', function () {
                        let targetNode = totalNodeList[this.getAttribute("Nodepos")];
                        let parameter = this.getAttribute("Parameter");
                        if (parameter == "input dimension" || parameter == "output dimension") {
                            targetNode.parameterList[parameter] = this.value.split(',').map(Number);
                        }
                        else {
                            targetNode.parameterList[parameter] = this.value;
                        }
                    })
                    parainput.placeholder = "Example:" + paraExample[key];
                    if (targetNode.parameterList[key]) {
                        if (key == "input dimension" || key == "output dimension") {
                            parainput.value = targetNode.parameterList[key].join(',');
                        }
                        else {
                            parainput.value = targetNode.parameterList[key];
                        }
                    }
                    pararow.classList.add('list-group-item');
                    pararow.innerHTML = key + ':';
                    pararow.appendChild(parainput);
                    paracardList.appendChild(pararow);
                })

            }
        }
        element.appendChild(image);
        var dot = document.createElement("div");
        dot.classList.add("dot");
        dot.classList.add("top")
        element.appendChild(dot)
        var dot = document.createElement("div");
        dot.classList.add("dot");
        dot.classList.add("bottom");
        element.appendChild(dot)
        var dot = document.createElement("div");
        dot.classList.add("dot");
        dot.classList.add("left");
        element.appendChild(dot)
        var dot = document.createElement("div");
        dot.classList.add("dot");
        dot.classList.add("right");
        element.appendChild(dot)
        element.style.left = (event.clientX - document.getElementById("canvas").getBoundingClientRect().left + Number(offX)) + 'px'; // 调整元件位置
        element.style.top = (event.clientY - document.getElementById("canvas").getBoundingClientRect().top + Number(offY)) + 'px';
        element.addEventListener('mouseover', function () {
            element.classList.add('image-hovered');
            for (i = 1; i < 5; i++) {
                element.children[i].style.visibility = 'visible';
            }
        });

        element.addEventListener('mouseout', function () {
            element.classList.remove('image-hovered');
            for (i = 1; i < 5; i++) {
                element.children[i].style.visibility = 'hidden';
            }
        });

        for (i = 1; i < 5; i++) {
            element.children[i].addEventListener('click', function (event) {
                dotclick_handler(event);
            });
        }


        event.target.parentNode.appendChild(element);
        let NewGraph = new NetGraph(GraphName);
        myGraph.addNode(NewGraph);
        totalNodeList.push(NewGraph);
        element.setAttribute("Nodepos", totalNodeList.indexOf(NewGraph));

    }
    else {
        var element = document.getElementById(id);
        element.style.left = (event.clientX - document.getElementById("canvas").getBoundingClientRect().left + Number(offX)) + 'px'; // 调整元件位置
        element.style.top = (event.clientY - document.getElementById("canvas").getBoundingClientRect().top + Number(offY)) + 'px';
        updateArrow(element);
        isDarggingElement = false
    }


}


function dragstart_handler(event) {

    console.log("Startdrag");
    isDarggingElement = true;
    var offX = event.target.getBoundingClientRect().left - event.clientX;
    var offY = event.target.getBoundingClientRect().top - event.clientY;
    event.dataTransfer.setData('offX', offX);
    event.dataTransfer.setData('offY', offY);
    event.dataTransfer.setData('id', 0);
    event.dataTransfer.setData('GraphName', event.target.getAttribute('imgname'));
}

function buttonClickHandler() {
    let paracard = document.getElementById("paracard");
    paracard.style.right = "-20%";

}

