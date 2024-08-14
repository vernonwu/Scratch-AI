class NetGraph {
    constructor(graphName) {
        this.nodeList = [];
        this.edgeList = [];
        this.inputDim = [];
        this.outputDim = [];
        this.parameterList = {};
        this.graphName = graphName;
        this.inputNode = null;
        this.outputNode = null;
    }

    addNode(Node) {
        if (!this.nodeList.includes(Node)) {
            this.nodeList.push(Node);
            let index = this.nodeList.indexOf(Node);
            this.edgeList = this.augmentMatrix(this.edgeList,index);
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

    deleteNode(Node){
        if (!this.nodeList.includes(Node)) {
            return false;
        }
        else {
            let index = this.nodeList.indexOf(Node);
            this.nodeList.splice(index,1);
            this.nodeList=this.augmentMatrix2(this.edgeList,index);
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
        this.inputDim = inputDim;
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

    augmentMatrix(matrix, pos,) {
        const len = matrix.length;
        for (let row= 0; row < len; row++) {
            matrix[row].splice(pos, 0, 0);
        }
        let newRow = Array(len + 1).fill(0);
        matrix.splice(pos, 0, newRow);
        return matrix;
    }

    augmentMatrix2(matrix, pos) {
        const len = matrix.length;
        for (let row = 0; row < len; row++) {
            console.log(matrix[row])
            matrix[row].splice(pos, 1);
        }
        matrix.splice(pos, 1);
        return matrix;
    }

    checkMatrix(matrix){
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
            return [zeroColIndex,zeroRowIndex];
        } else {
            return false;
        }
    };


    printGraph_bfs(matrix, namelist, startpos, endpos){
        let paths = [];
        let queue = [[startpos]];

        while(queue.length>0){
            let path = queue.shift();
            let node = path[path.length-1];

            if(node === endpos){
                paths.push(path);
            }

            for(let i = 0; i<matrix[node].length; i++){
                if(matrix[node][i] === 1 && !path.includes(i)){
                    let new_path = path.slice();
                    new_path.push(i);
                    queue.push(new_path);
                }
            }
        }
        for (let path of paths) {
            let pathNames = path.map(node => namelist[node]);
            console.log(pathNames.join('->'));
        }
    }
    
}

// let graph = new NetGraph("test");
// let FC = new NetGraph('FC');
// let pooling = new NetGraph('pooling');
// FC.setDim([64, 1], [10, 1])
// let Conv2d = new NetGraph('Conv2d');
// Conv2d.setDim([3,600,600],[24,10,10]);

// graph.addNode(FC);
// graph.addNode(Conv2d);
// graph.addNode(pooling);
// graph.addEdge(Conv2d,pooling);
// graph.addEdge(pooling,FC);
// graph.deleteNode(pooling);
// console.log(graph.edgeList)
// // namelist = ['FC','Conv2d','pooling']
// // let a,b;
// // [a,b] = graph.checkMatrix(graph.edgeList);
// // graph.printGraph_bfs(graph.edgeList,namelist,a,b);

let a = {sdsd:1, sdu:2};
let keys = Object.keys(a);
keys.forEach(element => {
    console.log(element)
});


