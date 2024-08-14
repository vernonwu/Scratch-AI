class NetGraph {
    constructor(graphName) {
        this.nodeList = [];
        this.edgeList = [];
        this.parameterList = {};
        this.graphName = graphName;
        this.initializeParameterList();
    }

    initializeParameterList() {
        this.parameterList["input dimension"] = null;
        this.parameterList["output dimension"] = null;
        if (this.graphName == 'conv2d') {
            this.parameterList["Activate Function"] = null;
            this.parameterList.padding = null;
            this.parameterList.kernel_size = null;
            this.parameterList.stride = null;
        }
        else if(this.graphName == 'FC'){
            this.parameterList["Activate Function"] = null;
        }
        else if(this.graphName =="pooling"){
            this.parameterList["Pooling type"] = null;
            this.parameterList.kernel_size = null;
            this.parameterList.stride = null;
        }
    }

    addNode(Node) {
        if (!this.nodeList.includes(Node)) {
            this.nodeList.push(Node);
            let index = this.nodeList.indexOf(Node);
            this.edgeList = this.augmentMatrix(this.edgeList, index);
            return true;
        }
        else {
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
        }
        else {
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

    setDim(inputDim, outputDim) {
        this.parameterList["input dimension"] = inputDim;
        this.parameterList["output dimension"] = outputDim;
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

    augmentMatrix(matrix, pos,) {
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
    };

    printGraph_bfsHelp(matrix, namelist, startpos, endpos) {
        let paths = [];
        let queue = [[startpos]];
        let condeAera = document.getElementById("code-area")

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
        condeAera.innerHTML = '';
        for (let path of paths) {
            let pathNames = path.map(node => namelist[node]);
            condeAera.innerHTML = condeAera.innerHTML +pathNames.join('->')+"\n\n";
            console.log(pathNames.join('->'));
        }
    }


    printGraph_bfs() {
        let condeAera = document.getElementById("code-area")
        if (this.checkMatrix(this.edgeList) === false) {
            condeAera.innerHTML = "Graph is incpomleted";
            console.log("Graph is incpomleted");
        }
        else {
            let GraphStartNode, GraphEndNode;
            [GraphStartNode, GraphEndNode] = this.checkMatrix(this.edgeList);
            let namelist = [];
            this.nodeList.forEach(node => {
                namelist.push(node.graphName+JSON.stringify(node.parameterList));
            });
            this.printGraph_bfsHelp(this.edgeList, namelist, GraphStartNode, GraphEndNode);
        }


    }

}


var globalclicktime = 0;
let arrowId = 0;
let componentWidth = 120;
let componentHeight = 180;
var tool = 0;
let myGraph = new NetGraph('Main-Graph');
let totalNodeList = [];
let paraExample = { 'kernel_size': 3, "padding": 0, "stride": 1, "input dimension": "3,500,800", "output dimension": "3,500,800" }
const ActivateFunctionList = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'ELU', 'Softmax', 'Swish'];
const PoolingTypeList = ['Max',"Min","Average"];

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
        var pathData = line.getAttribute("d").split(" ");
        var shiftY = parseFloat(line.getAttribute("shiftY"));
        pathData[4] = (parseFloat(pathData[1]) + offsetX) / 2;
        pathData[5] = (parseFloat(pathData[2]) + offsetY) / 2 + shiftY + sign(shiftY) * Math.abs(offsetY - parseFloat(pathData[2]));
        pathData[6] = offsetX;
        pathData[7] = offsetY;
        line.setAttribute("d", pathData.join(" "));
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
        cardimg.src = "image/" + image.getAttribute('imgname') + '.png';
        let cardtitle = document.getElementById('cardtitle')
        cardtitle.textContent = image.getAttribute('imgname');
    });

    image.addEventListener('mouseout', function (event) {
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

function GenerateToolOnclick(event){
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
            line.setAttribute("d", "M " + offsetX + " " + offsetY + " Q " + offsetX + " " + offsetY + " " + offsetX + " " + offsetY);
            line.setAttribute("stroke", "black"); // 设置描边颜色
            line.setAttribute("fill", "transparent");
            line.setAttribute('marker-end', 'url(#arrow)');
            line.setAttribute('marker-start', 'url(#circle)')
            line.setAttribute("stroke-width", 1);
            line.id = arrowId;
            line.setAttribute("shiftY", Math.floor(-event.target.parentNode.getBoundingClientRect().height / 2 - event.target.parentNode.getBoundingClientRect().top + event.target.getBoundingClientRect().top) + 4);
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
        var pathData = line.getAttribute("d").split(" ");
        var shiftY = parseFloat(line.getAttribute("shiftY"));
        pathData[4] = (parseFloat(pathData[1]) + offsetX) / 2;
        pathData[5] = (parseFloat(pathData[2]) + offsetY) / 2 + shiftY + sign(shiftY) * Math.abs(offsetY - parseFloat(pathData[2]));
        pathData[6] = offsetX;
        pathData[7] = offsetY;
        line.setAttribute("d", pathData.join(" "));
        line.classList.add("line");
        myGraph.addEdge(startNode, endNode);
        arrowId++;
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
                if (arrowList[2 * j + 1] == "start") {
                    line = document.getElementById(arrowList[2 * j])
                    var pathData = line.getAttribute("d").split(" ");
                    var shiftY = parseFloat(line.getAttribute("shiftY"));
                    pathData[1] = offsetX;
                    pathData[2] = offsetY;
                    pathData[5] = (parseFloat(pathData[2]) + parseFloat(pathData[7])) / 2 + shiftY + sign(shiftY) * Math.abs(parseFloat(pathData[7]) - parseFloat(pathData[2]));
                    line.setAttribute("d", pathData.join(" "));
                }
                else if (arrowList[2 * j + 1] == "end") {
                    line = document.getElementById(arrowList[2 * j])
                    var pathData = line.getAttribute("d").split(" ");
                    var shiftY = parseFloat(line.getAttribute("shiftY"));
                    pathData[6] = offsetX;
                    pathData[7] = offsetY;
                    pathData[5] = (parseFloat(pathData[2]) + parseFloat(pathData[7])) / 2 + shiftY + sign(shiftY) * Math.abs(parseFloat(pathData[7]) - parseFloat(pathData[2]));
                    line.setAttribute("d", pathData.join(" "));
                }
            }
        }
    }
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
        element.id = ran;
        element.setAttribute('GraphName', GraphName);
        element.style.width = componentWidth + 'px';
        element.style.height = componentHeight + 'px';
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
                        if(targetNode.parameterList[key]){
                            parainput.value=targetNode.parameterList[key];

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
                        if(targetNode.parameterList[key]){
                            parainput.value=targetNode.parameterList[key];

                        }

                        pararow.innerHTML = key + ':';
                        pararow.appendChild(parainput);
                        paracardList.appendChild(pararow);
                        return;
                    }
                    
                    var pararow = document.createElement('li');
                    var parainput = document.createElement('input');
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
                    if(targetNode.parameterList[key]){
                        if(key == "input dimension" || key == "output dimension"){
                            parainput.value=targetNode.parameterList[key].join(',');
                        }
                        else{
                            parainput.value=targetNode.parameterList[key];
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

function buttonClickHandler(){
    let paracard = document.getElementById("paracard");
    paracard.style.right="-20%";

}

