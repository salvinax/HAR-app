import * as THREE from "three"
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import * as dat from 'dat.gui'

let scene, renderer, camera, clock, mixer, model
let currentAction = '0Tpose'
let actionsList = []
let img = document.createElement('img');
let box = document.getElementById('floating-box')
box.appendChild(img)
let flag = 0
let button = document.getElementById('close-box')

button.addEventListener("click", function () {
    if (flag) {
        box.style.top = '-28px'
        button.innerHTML = 'Close Graphs'
        flag = 0
    } else {
        box.style.top = '-320px'
        button.innerHTML = 'Open Graphs'
        flag = 1
    }
})

init()

window.addEventListener('DOMContentLoaded', () => {
    const websocket = new WebSocket("ws://localhost:8001/")
    receivePredictions(websocket)
})

window.addEventListener('resize', function () {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)

})


function init() {

    renderer = new THREE.WebGLRenderer()
    renderer.shadowMap.enabled = true
    renderer.setSize(window.innerWidth, window.innerHeight)
    document.getElementById('canvas').appendChild(renderer.domElement)

    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000)

    scene.background = new THREE.Color(0xa0a0a0);
    scene.fog = new THREE.Fog(0xa0a0a0, 10, 50);


    clock = new THREE.Clock()
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x8d8d8d, 3);
    hemiLight.position.set(0, 20, 0);
    scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 3);
    dirLight.position.set(-3, 10, 10);
    dirLight.castShadow = true;
    dirLight.shadow.camera.top = 5;
    dirLight.shadow.camera.bottom = - 2;
    dirLight.shadow.camera.left = 0;
    dirLight.shadow.camera.right = 5;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 40;
    scene.add(dirLight);


    // const dLightShadowHelper = new THREE.CameraHelper(dirLight.shadow.camera)
    // scene.add(dLightShadowHelper)


    const orbit = new OrbitControls(camera, renderer.domElement)
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), new THREE.MeshPhongMaterial({ color: 0xcbcbcb, depthWrite: false }));
    mesh.rotation.x = - Math.PI / 2;
    mesh.receiveShadow = true;
    scene.add(mesh);

    // const axesHelper = new THREE.AxesHelper(3)
    // scene.add(axesHelper)

    camera.position.set(-3, 2.5, 11)
    orbit.update()

    // const grid = new THREE.GridHelper(0, 0)
    // scene.add(grid)
    const characterUrl = new URL('../assets/3dCharacter.glb', import.meta.url)
    const assetLoader = new GLTFLoader()


    assetLoader.load(characterUrl.href, function (gltf) {

        model = gltf.scene
        console.log(model)

        scene.add(model)

        // change material of model
        const material = new THREE.MeshPhongMaterial({ color: 0xC0C076 }); // Set your desired color
        model.traverse((child) => {
            if (child.isMesh) {
                child.material = material;
                child.castShadow = true
            }
        })


        model.position.set(1, 0, 3)

        // load all actions into array 
        mixer = new THREE.AnimationMixer(model) //animation player
        const animations = gltf.animations;
        animations.forEach(element => {
            const action = mixer.clipAction(element)
            actionsList[element.name] = action
        });
        // default animation 
        actionsList['0Tpose'].play()

        // const clip = THREE.AnimationClip.findByName(animations, 'WALKING')
        // const action = mixer.clipAction(clip)
        // action.setEffectiveTimeScale(0.6)
        // action.play();

    }, undefined, function (error) {
        console.log(error)
    })

    const gui = new dat.GUI()
    const options = {
        modelColor: '#C0C076',
    }

    gui.addColor(options, 'modelColor').onChange(function (e) {
        model.traverse((child) => {
            if (child.isMesh) {
                child.material.color.set(e);
            }
        })
    })



}

function receivePredictions(websocket) {
    websocket.addEventListener("message", ({ data }) => {
        const event = JSON.parse(data);

        if (event) {
            if (event.type == "pred") {
                if (event.result == 0) {
                    console.log('we walking')
                    playAnimation('WALKING')
                } else if (event.result == 1) {
                    console.log('we jumping')
                    playAnimation('JUMP')
                }
            } else if (event.type == "image") {
                const encoded_img = 'data:image/png;base64,' + event.img
                img.src = encoded_img
            }
        } else {
            playAnimation('0Tpose')
        }
    });
}

function playAnimation(action) {
    let newAction = actionsList[action]
    let currAction = actionsList[currentAction]

    // if (newAction !== currAction) {
    //     // Crossfade to the new action
    //     currAction.crossFadeTo(newAction, 0.1, true);
    //     newAction.play()
    // }

    // currentAction = action;

    if (newAction !== currAction) {
        if (currAction) {
            currAction.stop()
            newAction.play()
            currentAction = action
            // Play the new action without crossfade if there is no current action
        }
    }
}



function animate() {


    if (mixer)
        mixer.update(clock.getDelta())
    renderer.render(scene, camera)
    //requestAnimationFrame(animate)
    //animate()
}


renderer.setAnimationLoop(animate)

