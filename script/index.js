var globalclicktime = 0;
let arrowId = 0;
let componentId = 10000;
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
    let code = myGraph.printGraph_bfs();
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
        element.id = componentId;
        componentId++;
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
        console.log(element);
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

