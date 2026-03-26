package com.padFrontend

import androidx.activity.compose.rememberLauncherForActivityResult
import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewModelScope
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.okhttp.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.cookies.*
import io.ktor.client.plugins.websocket.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.websocket.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

// ─────────────────────────────────────────────────────────────────────────────
// DATA MODELS — inchangés
// ─────────────────────────────────────────────────────────────────────────────

@Serializable
data class SessionData(val url: String, val challenge: List<String>)

// ─────────────────────────────────────────────────────────────────────────────
// DESIGN SYSTEM — Swiss / Red & Black
// ─────────────────────────────────────────────────────────────────────────────

val SwissWhite   = Color(0xFFFFFFFF)
val SwissBlack   = Color(0xFF000000)
val SwissRed     = Color(0xFFFF0000)
val SwissGray    = Color(0xFFF5F5F5)
val SwissGrayMid = Color(0xFFE0E0E0)
val SwissGrayDim = Color(0xFF9E9E9E)

val directionArrow = mapOf(
    "HAUT"   to "↑",
    "BAS"    to "↓",
    "GAUCHE" to "←",
    "DROITE" to "→"
)

// ─────────────────────────────────────────────────────────────────────────────
// NETWORK CLIENT — inchangé
// ─────────────────────────────────────────────────────────────────────────────

object PadClient {
    val client = HttpClient(OkHttp) {
        install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) }
        install(WebSockets)
        install(HttpCookies) { storage = AcceptAllCookiesStorage() }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VIEWMODEL — inchangé
// ─────────────────────────────────────────────────────────────────────────────

class PadViewModel : ViewModel() {
    var step         by mutableStateOf("IDLE")
    var challenge    by mutableStateOf<List<String>>(emptyList())
    var resultStatus by mutableStateOf("")
    var uploadUrl    = ""
    var filename     = ""

    fun startSession() {
        viewModelScope.launch {
            step = "CONNECTING"
            try {
                val authRes = PadClient.client.get("http://localhost:8080/Authentificate")
                if (authRes.status == HttpStatusCode.OK) {
                    val data: SessionData = PadClient.client.get("http://localhost:8080/init-sesssion").body()
                    challenge = data.challenge
                    uploadUrl = data.url
                    filename  = uploadUrl.substringAfterLast("/").substringBefore("?")
                    step = "READY"
                }
            } catch (e: Exception) { step = "ERROR" }
        }
    }

    fun uploadAndListen(file: File) {
        viewModelScope.launch {
            step = "MIRRORING"
            try {
                val res = PadClient.client.put(uploadUrl) {
                    setBody(file.readBytes())
                    header(HttpHeaders.ContentType, "video/mp4")
                }
                if (res.status == HttpStatusCode.OK || res.status == HttpStatusCode.Created) {
                    step = "DEEPFAKE"
                    PadClient.client.webSocket("ws://localhost:8080/ws/$filename") {
                        send("UploadDone|${challenge.joinToString(",")}")
                        for (frame in incoming) {
                            if (frame is Frame.Text) {
                                val msg = frame.readText()
                                when {
                                    msg.contains("DEEPFAKE_OK") -> {
                                        step = "DEEPFAKE_OK"
                                        delay(2000)
                                        step = "PROCESSING"
                                    }
                                    msg.contains("IA_SUCCESS") -> { resultStatus = "SUCCESS"; step = "RESULT" }
                                    msg.contains("IA_FAIL")    -> { resultStatus = "FAIL";    step = "RESULT" }
                                    msg.contains("ERROR")      -> { resultStatus = "ERROR";   step = "RESULT" }
                                }
                                if (step == "RESULT") break
                            }
                        }
                    }
                }
            } catch (e: Exception) { step = "ERROR" }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPOSANTS UTILITAIRES
// ─────────────────────────────────────────────────────────────────────────────

/** Ligne de séparation fine noir */
@Composable
fun SwissDivider() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(SwissBlack)
    )
}

/** Divider gris léger */
@Composable
fun SwissDividerLight() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(SwissGrayMid)
    )
}

/** Bouton primaire noir */
@Composable
fun PrimaryButton(label: String, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .background(SwissBlack)
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Text(
            label,
            color = SwissWhite,
            fontSize = 12.sp,
            fontWeight = FontWeight.Black,
            letterSpacing = 3.sp
        )
    }
}

/** Bouton outline */
@Composable
fun OutlineButton(label: String, color: Color = SwissBlack, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .border(1.dp, color)
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Text(
            label,
            color = color,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 3.sp
        )
    }
}

/** Label technique uppercase gris */
@Composable
fun TechLabel(text: String) {
    Text(
        text,
        color = SwissGrayDim,
        fontSize = 9.sp,
        fontWeight = FontWeight.Medium,
        letterSpacing = 2.sp
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// APP SHELL
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun PadApp(vm: PadViewModel = viewModel()) {
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ -> }
    LaunchedEffect(Unit) {
        launcher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
    }

    val recorder     = remember { Recorder.Builder().setQualitySelector(QualitySelector.from(Quality.LOWEST)).build() }
    val videoCapture = remember { VideoCapture.withOutput(recorder) }

    Box(modifier = Modifier.fillMaxSize().background(SwissWhite)) {

        // Caméra en fond (états READY / MIRRORING)
        AnimatedVisibility(
            visible = vm.step in listOf("READY", "CONNECTING", "MIRRORING"),
            enter = fadeIn(tween(400)),
            exit  = fadeOut(tween(300))
        ) {
            CameraLayer(videoCapture)
            // Overlay blanc semi-transparent pour garder la lisibilité des éléments
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(SwissWhite.copy(alpha = 0.55f))
            )
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp)
                .padding(top = 56.dp, bottom = 40.dp)
        ) {
            // Header permanent
            AppHeader(step = vm.step)

            Spacer(modifier = Modifier.height(8.dp))
            SwissDivider()

            // Contenu central
            Box(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                AnimatedContent(
                    targetState = vm.step,
                    transitionSpec = { fadeIn(tween(300)) togetherWith fadeOut(tween(200)) },
                    label = "stepContent"
                ) { s ->
                    when (s) {
                        "IDLE"        -> IdleView()
                        "CONNECTING"  -> ConnectingView()
                        "DEEPFAKE"    -> DeepfakeCheckingView()
                        "DEEPFAKE_OK" -> DeepfakeOkView()
                        "PROCESSING"  -> ProcessingView()
                        "RESULT"      -> ResultView(vm.resultStatus) { vm.step = "IDLE" }
                        "ERROR"       -> ErrorView { vm.startSession() }
                        else          -> {}
                    }
                }
            }

            SwissDivider()
            Spacer(modifier = Modifier.height(24.dp))

            // Actions bas de page
            AnimatedVisibility(
                visible = vm.step == "IDLE",
                enter = fadeIn(tween(400)),
                exit  = fadeOut(tween(200))
            ) {
                PrimaryButton("S'IDENTIFIER BIOMÉTRIQUEMENT") { vm.startSession() }
            }

            AnimatedVisibility(
                visible = vm.step == "READY",
                enter = fadeIn(tween(400)),
                exit  = fadeOut(tween(200))
            ) {
                ChallengeUI(vm.challenge, videoCapture) { file -> vm.uploadAndListen(file) }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HEADER
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun AppHeader(step: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.Bottom
    ) {
        Column {
            Text(
                "PAD",
                color = SwissBlack,
                fontSize = 28.sp,
                fontWeight = FontWeight.Black,
                letterSpacing = 8.sp
            )
            Text(
                "PRÉSENCE AUTHENTIFICATION",
                color = SwissGrayDim,
                fontSize = 8.sp,
                letterSpacing = 2.sp
            )
        }
        // Indicateur d'état
        val stateColor = when (step) {
            "READY", "RESULT" -> SwissRed
            "ERROR"           -> SwissGrayDim
            else              -> SwissBlack
        }
        Text(
            step,
            color = stateColor,
            fontSize = 9.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 2.sp
        )
    }
    Spacer(modifier = Modifier.height(16.dp))
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE IDLE — page de garde magazine
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun IdleView() {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.Start
    ) {
        // Grand numéro d'édition
        Text(
            "01",
            color = SwissGray,
            fontSize = 120.sp,
            fontWeight = FontWeight.Black,
            lineHeight = 100.sp,
            letterSpacing = (-2).sp
        )

        Spacer(modifier = Modifier.height((-16).dp))

        Text(
            "BIOMETRIC\nIDENTITY\nVERIFICATION",
            color = SwissBlack,
            fontSize = 36.sp,
            fontWeight = FontWeight.Black,
            lineHeight = 38.sp,
            letterSpacing = (-0.5).sp
        )

        Spacer(modifier = Modifier.height(24.dp))
        SwissDividerLight()
        Spacer(modifier = Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            TechLabel("SYSTÈME V2.4")
            TechLabel("ENCLAVE SÉCURISÉE")
            TechLabel("ISO 30107-3")
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            "Placez-vous face à la caméra et effectuez la séquence gestuelle indiquée pour confirmer votre présence.",
            color = SwissGrayDim,
            fontSize = 13.sp,
            fontWeight = FontWeight.Light,
            lineHeight = 20.sp
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE CONNEXION
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ConnectingView() {
    val progress by rememberInfiniteTransition(label = "progress").animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1400, easing = LinearEasing)),
        label = "bar"
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            "CONNEXION",
            color = SwissBlack,
            fontSize = 32.sp,
            fontWeight = FontWeight.Black,
            letterSpacing = 4.sp
        )
        Spacer(modifier = Modifier.height(8.dp))
        TechLabel("ÉTABLISSEMENT DU CANAL SÉCURISÉ")
        Spacer(modifier = Modifier.height(40.dp))

        // Barre de progression minimale
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(SwissGrayMid)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(progress)
                    .background(SwissBlack)
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE CHALLENGE (READY)
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ChallengeUI(
    challenge: List<String>,
    videoCapture: VideoCapture<Recorder>,
    onReady: (File) -> Unit
) {
    Column {
        // Directions — typographie massive
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            challenge.forEachIndexed { i, dir ->
                val arrow = directionArrow[dir] ?: "?"
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        "${i + 1}",
                        color = SwissRed,
                        fontSize = 9.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = 1.sp
                    )
                    Text(
                        arrow,
                        color = SwissBlack,
                        fontSize = 32.sp,
                        fontWeight = FontWeight.Black
                    )
                    Text(
                        dir,
                        color = SwissGrayDim,
                        fontSize = 7.sp,
                        letterSpacing = 1.sp
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
        RecordButton(videoCapture, onReady)
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            "Maintenez pour enregistrer · Relâchez pour analyser",
            color = SwissGrayDim,
            fontSize = 9.sp,
            letterSpacing = 0.5.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE ANTI-DEEPFAKE — analyse en cours
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun DeepfakeCheckingView() {
    val progress by rememberInfiniteTransition(label = "df").animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1200, easing = LinearEasing)),
        label = "bar"
    )
    Column(
        horizontalAlignment = Alignment.Start,
        modifier = Modifier.fillMaxWidth()
    ) {
        Box(modifier = Modifier.width(40.dp).height(4.dp).background(SwissBlack))
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "ANALYSE\nANTI-DEEPFAKE",
            color = SwissBlack,
            fontSize = 42.sp,
            fontWeight = FontWeight.Black,
            lineHeight = 46.sp,
            letterSpacing = (-0.5).sp
        )
        Spacer(modifier = Modifier.height(24.dp))
        SwissDividerLight()
        Spacer(modifier = Modifier.height(16.dp))
        TechLabel("DÉTECTION DE SYNTHÈSE IA · VÉRIFICATION BIOMÉTRIQUE PASSIVE")
        Spacer(modifier = Modifier.height(32.dp))
        Box(
            modifier = Modifier.fillMaxWidth().height(1.dp).background(SwissGrayMid)
        ) {
            Box(
                modifier = Modifier.fillMaxHeight().fillMaxWidth(progress).background(SwissBlack)
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE ANTI-DEEPFAKE — résultat OK
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun DeepfakeOkView() {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.Start
    ) {
        Box(modifier = Modifier.width(40.dp).height(4.dp).background(SwissBlack))
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "ANALYSE\nRÉUSSIE",
            color = SwissBlack,
            fontSize = 52.sp,
            fontWeight = FontWeight.Black,
            lineHeight = 54.sp,
            letterSpacing = (-1).sp
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            "Vidéo est réelle — aucun deepfake détecté",
            color = SwissGrayDim,
            fontSize = 13.sp,
            fontWeight = FontWeight.Light
        )
        Spacer(modifier = Modifier.height(32.dp))
        SwissDividerLight()
        Spacer(modifier = Modifier.height(16.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Filled.CheckCircle,
                contentDescription = null,
                tint = SwissBlack,
                modifier = Modifier.size(16.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            TechLabel("AUTHENTICITÉ CONFIRMÉE · PASSAGE À L'ANALYSE GESTUELLE")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE PROCESSING
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ProcessingView() {
    val stepF by rememberInfiniteTransition(label = "step").animateFloat(
        initialValue = 0f, targetValue = 3f,
        animationSpec = infiniteRepeatable(tween(1800, easing = LinearEasing)),
        label = "dots"
    )
    val dots = ".".repeat(stepF.toInt().coerceIn(0, 3) + 1)

    Column(
        horizontalAlignment = Alignment.Start,
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            "ANALYSE\nEN COURS$dots",
            color = SwissBlack,
            fontSize = 42.sp,
            fontWeight = FontWeight.Black,
            lineHeight = 46.sp,
            letterSpacing = (-0.5).sp
        )
        Spacer(modifier = Modifier.height(24.dp))
        SwissDividerLight()
        Spacer(modifier = Modifier.height(16.dp))
        TechLabel("TRAITEMENT PAR IA NEURALE · NE PAS BOUGER")

        Spacer(modifier = Modifier.height(32.dp))
        LinearProgressIndicator(
            modifier = Modifier.fillMaxWidth().height(1.dp),
            color = SwissBlack,
            trackColor = SwissGrayMid
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE RÉSULTAT
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ResultView(status: String, onReset: () -> Unit) {
    val isSuccess = status == "SUCCESS"
    val accent    = if (isSuccess) SwissBlack else SwissRed
    val headline  = if (isSuccess) "ACCÈS\nAUTORISÉ" else "ACCÈS\nREFUSÉ"
    val sub       = if (isSuccess) "Identité biométrique confirmée" else "Séquence non reconnue"

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.Start
    ) {
        // Indicateur couleur
        Box(
            modifier = Modifier
                .width(40.dp)
                .height(4.dp)
                .background(accent)
        )
        Spacer(modifier = Modifier.height(16.dp))

        Text(
            headline,
            color = accent,
            fontSize = 52.sp,
            fontWeight = FontWeight.Black,
            lineHeight = 54.sp,
            letterSpacing = (-1).sp
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            sub,
            color = SwissGrayDim,
            fontSize = 13.sp,
            fontWeight = FontWeight.Light
        )

        Spacer(modifier = Modifier.height(40.dp))
        SwissDividerLight()
        Spacer(modifier = Modifier.height(24.dp))

        OutlineButton("TERMINER SESSION", color = accent) { onReset() }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGE ERREUR
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ErrorView(onRetry: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.Start
    ) {
        Box(
            modifier = Modifier
                .width(40.dp)
                .height(4.dp)
                .background(SwissRed)
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            "ERREUR\nSYSTÈME",
            color = SwissBlack,
            fontSize = 52.sp,
            fontWeight = FontWeight.Black,
            lineHeight = 54.sp
        )
        Spacer(modifier = Modifier.height(8.dp))
        TechLabel("CONNEXION AU SERVEUR IMPOSSIBLE")
        Spacer(modifier = Modifier.height(40.dp))
        SwissDividerLight()
        Spacer(modifier = Modifier.height(24.dp))
        OutlineButton("RÉESSAYER", color = SwissRed) { onRetry() }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BOUTON D'ENREGISTREMENT
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun RecordButton(videoCapture: VideoCapture<Recorder>, onFileReady: (File) -> Unit) {
    val context = LocalContext.current
    var isRecording          by remember { mutableStateOf(false) }
    var currentRecording: Recording? by remember { mutableStateOf(null) }
    val interactionSource = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }

    val bgColor    = if (isRecording) SwissRed   else SwissBlack
    val labelColor = SwissWhite
    val label      = if (isRecording) "■  ARRÊTER" else "●  ENREGISTRER"

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(56.dp)
            .background(bgColor)
            .clickable(interactionSource = interactionSource, indication = null) {
                if (isRecording) {
                    currentRecording?.stop()
                    isRecording = false
                } else {
                    val file = File(context.cacheDir, "pad_auth.mp4")
                    currentRecording = videoCapture.output
                        .prepareRecording(context, FileOutputOptions.Builder(file).build())
                        .start(ContextCompat.getMainExecutor(context)) { event ->
                            if (event is VideoRecordEvent.Finalize) onFileReady(file)
                        }
                    isRecording = true
                }
            },
        contentAlignment = Alignment.Center
    ) {
        Text(
            label,
            color = labelColor,
            fontSize = 12.sp,
            fontWeight = FontWeight.Black,
            letterSpacing = 3.sp
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// COUCHE CAMÉRA — inchangée
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun CameraLayer(videoCapture: VideoCapture<Recorder>) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView    = remember { PreviewView(context) }

    LaunchedEffect(Unit) {
        val cameraProvider = ProcessCameraProvider.getInstance(context).get()
        val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_FRONT_CAMERA,
            preview,
            videoCapture
        )
    }
    AndroidView(
        factory  = { previewView },
        modifier = Modifier.fillMaxSize()
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// ENTRY POINT — inchangé
// ─────────────────────────────────────────────────────────────────────────────

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { PadApp() } }
    }
}