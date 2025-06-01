import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { Line2 } from 'three/addons/lines/Line2.js';


const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
renderer.setAnimationLoop( animate );
document.body.appendChild( renderer.domElement );

const clock = new THREE.Clock();

camera.position.z = 1.0;
camera.position.y = 1.7;
camera.position.x = 1.0;
camera.lookAt( new THREE.Vector3(0, 1, 0) );

const controls = new OrbitControls( camera, renderer.domElement );
controls.target.set( 0, 1, 0 );
controls.update();
controls.enablePan = false;
controls.enableDamping = true;


const color = 0xFFFFFF;
const intensity = 1;
const light = new THREE.AmbientLight(color, intensity);
scene.add(light);

async function loadData() {
    const response = await fetch("./drone_data.json");
    const data = await response.json();

    return data;
}

const planeSize = 40;
const loader = new THREE.TextureLoader();
const texture = loader.load('checker.png');
texture.wrapS = THREE.RepeatWrapping;
texture.wrapT = THREE.RepeatWrapping;
texture.magFilter = THREE.NearestFilter;
texture.colorSpace = THREE.SRGBColorSpace;
const repeats = planeSize / 2;
texture.repeat.set(repeats, repeats);
const planeGeo = new THREE.PlaneGeometry(planeSize, planeSize);
const planeMat = new THREE.MeshPhongMaterial({
    map: texture,
    side: THREE.DoubleSide,
});
const mesh = new THREE.Mesh(planeGeo, planeMat);
mesh.rotation.x = Math.PI * -.5;
scene.add(mesh);

var mixers = [];
const gltfLoader = new GLTFLoader();
var droneObject = undefined;
gltfLoader.load( '/drone6.glb', function ( gltf ) {
    gltf.scene.scale.x = 2;
    gltf.scene.scale.y = 2;
    gltf.scene.scale.z = 2;
    console.log(gltf);
    droneObject = gltf.scene;
    scene.add(gltf.scene)

    // Play all animations
    const clips = gltf.animations;
    for (const clip of clips) {
        const name = clip.tracks[0].name.split(".")[0]
        const child = gltf.scene.children.find(x => x.name == name);
        const mixer = new THREE.AnimationMixer( child );
        mixer.clipAction(clip).play();
        mixers.push(mixer);
    }
}, undefined, function ( error ) {

  console.error( error );

} );


var trajectory = [[0, [0, 0, 1]]];
var velocities = [[0, [0, 0, 0]]];
function updateData() {
    loadData().then(data => {
        trajectory = data.flight_path;
        velocities = data.velocities;

        const points = [];
        for (const point of data.target_trajectory) {
            points.push(new THREE.Vector3(point[0], point[2], point[1]));
        }
        const geometry = new THREE.BufferGeometry().setFromPoints( points );
        const targetMaterial = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
        const line = new THREE.Line( geometry, targetMaterial );
        scene.add( line );
    });
}
updateData();
setInterval(updateData, 1000);

// flight path as line
var previousIndex = 0;
const MAX_POINTS = 100_000;
const flightPathGeometry = new THREE.BufferGeometry();
const flightPathPositions = new Float32Array( MAX_POINTS * 3 );
flightPathGeometry.setAttribute( 'position', new THREE.BufferAttribute( flightPathPositions, 3 ) );
flightPathGeometry.setDrawRange(0, 0);
const flightPathMaterial = new THREE.LineBasicMaterial({color: 0xff0000, linewidth: 10});
const flightPathLine = new THREE.Line(flightPathGeometry, flightPathMaterial);
scene.add(flightPathLine);
const flightPathPositionsAttribute = flightPathLine.geometry.getAttribute('position');


function animate() {
    const dt = clock.getDelta();
    const t = clock.getElapsedTime();
    let index = 0;
    while (index < trajectory.length && trajectory[index][0] <= t) {
        index += 1;
    }
    index -= 1;
    if (droneObject) {
        droneObject.position.x = trajectory[index][1][0];
        droneObject.position.y = trajectory[index][1][2];
        droneObject.position.z = trajectory[index][1][1];
        
        const v = new THREE.Vector3(velocities[index][1][0], velocities[index][1][2], velocities[index][1][1]);
        const vy = 0 + v.y;
        v.setY(0);
        for (const mixer of mixers) mixer.update( (5 + 3 * v.length() + 7 * vy) * dt );
        droneObject.rotation.z = - Math.PI / 4 * v.x;
        droneObject.rotation.x = Math.PI / 4 * v.z;
    }
    
    for (let i = previousIndex; i <= index; i++) {
        flightPathPositionsAttribute.setXYZ(i, trajectory[i][1][0], trajectory[i][1][2], trajectory[i][1][1])
    }
    flightPathPositionsAttribute.needsUpdate = true;
    previousIndex = index;
    flightPathLine.geometry.setDrawRange(0, index+1);
    renderer.render( scene, camera );
}
