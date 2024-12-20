
require('@tensorflow/tfjs');
const use = require('@tensorflow-models/universal-sentence-encoder');
const tf = require('@tensorflow/tfjs');


async function main() {

  const model = await use.loadQnA();

   // Base de conocimiento
   const responses = [
    'I am feeling great!',
    'The capital of China is Beijing.',
    'You have five fingers on your hand.',
    'The sky is blue on a clear day.',
    'Water boils at 100 degrees Celsius.',
    'Tengo 27 anios',
    'Soy de Guatemala',
    'Mi color favorito es el azul',
    'Mi comida favorita es la pizza',
    'Mi pelicula favorita es Star Wars',
    'Mi serie favorita es Breaking Bad',
    'Mi libro favorito es El Alquimista',
    'Mi deporte favorito es el futbol',
    'Mi equipo favorito es el Barcelona',
    'Mi animal favorito es el perro',
    'Mi pasatiempo favorito es leer',
    'Mi hobby favorito es cocinar',
    'Mi lugar favorito es la playa',
    'Mi cancion favorita es come as you are',
    'Mi artista favorito es Nirvana',
    'Mi grupo favorito es Nirvana',
    'Mi genero favorito es el rock',
    'Mi estilo favorito es el grunge',
    'Mi marca favorita es Nike',
    'Mi marca favorita es Adidas',
    'Mi marca favorita es Puma',
    'Mi marca favorita es Vans',
    "Hola, ¿cómo puedo ayudarte hoy?",
    "Lo siento, no entiendo lo que estás diciendo.",
    "Estoy aquí para responder tus preguntas.",
    "Eso suena interesante, ¿quieres saber más sobre eso?",
    "¿En qué más puedo ayudarte?",
    "El lenguaje de programación más utilizado es Python.",
    "La inteligencia artificial está cambiando el mundo rápidamente.",
    "El sistema operativo más popular es Windows.",
    "Los servidores en la nube son una parte fundamental de la tecnología moderna.",
    "La programación orientada a objetos es una de las principales metodologías de desarrollo.",
    "El monte Everest es la montaña más alta del mundo.",
    "Francia es famosa por la Torre Eiffel.",
    "La capital de España es Madrid.",
    "Brasil es conocido por sus hermosas playas y la selva amazónica.",
    "México tiene una rica historia y cultura.",
    "Albert Einstein es conocido por la teoría de la relatividad.",
    "El sol es una estrella que da luz y calor a la Tierra.",
    "El agua es esencial para la vida en la Tierra.",
    "La Mona Lisa es una de las obras más famosas de Leonardo da Vinci.",
    "La Revolución Francesa ocurrió a finales del siglo XVIII.",
    "Mi comida favorita es el sushi.",
    "Me encanta viajar por el mundo.",
    "Disfruto mucho de la música clásica.",
    "Mi deporte favorito es el tenis.",
    "Soy un fanático del cine de terror.",
    "Soy como un robot, pero un poco más divertido.",
    "¡Estoy tan feliz de hablar contigo hoy!",
    "¡Me encanta aprender cosas nuevas cada día!",
    "Soy una inteligencia artificial, pero también tengo sentido del humor.",
    "El cielo es el límite, ¡vamos a explorar todo el conocimiento!",
    "¡Hola! ¿En qué puedo ayudarte hoy?",
    "¡Qué bueno que me contactes, cuéntame qué necesitas saber!",
    "No estoy seguro de eso, pero puedo investigar más información.",
    "¿Te gustaría saber algo más sobre este tema?",
    "No entiendo bien lo que estás diciendo, ¿puedes repetirlo?",
    "Lo siento, no puedo ayudarte con eso en este momento.",
    "No tengo información suficiente sobre ese tema.",
    "No estoy seguro de la respuesta, pero puedo buscar más detalles.",
    "Esa no es una pregunta que pueda responder ahora.",
    "La gravedad es la fuerza que atrae los objetos hacia el centro de la Tierra.",
    "El cuerpo humano tiene 206 huesos.",
    "El ADN es la molécula que lleva la información genética.",
    "Los átomos están formados por protones, neutrones y electrones.",
    "El cambio climático está siendo impulsado por las actividades humanas.",
    "La Revolución Industrial comenzó en el siglo XVIII.",
    "Cristóbal Colón llegó a América en 1492.",
    "La Segunda Guerra Mundial ocurrió entre 1939 y 1945.",
    "Napoleón Bonaparte fue un líder militar francés muy conocido.",
    "La independencia de los Estados Unidos se proclamó en 1776.",
    "El fútbol es el deporte más popular del mundo.",
    "Usain Bolt es considerado el hombre más rápido del mundo.",
    "El baloncesto se juega con un balón y se intenta encestar en un aro.",
    "El tenis se juega con una pelota y una raqueta.",
    "El hockey sobre hielo es un deporte popular en países fríos como Canadá.",
    "El cine de ciencia ficción es uno de los géneros más populares.",
    "Los superhéroes están de moda gracias a las películas de Marvel y DC.",
    "Harry Potter es una de las sagas literarias más vendidas de la historia.",
    "La música pop es uno de los géneros musicales más escuchados en todo el mundo.",
    "La película 'Titanic' fue un éxito mundial en 1997.",
    "Sólo sé que no sé nada.",
    "La vida es aquello que te sucede mientras estás ocupado haciendo otros planes.",
    "El hombre es la medida de todas las cosas.",
    "La felicidad no es algo hecho. Viene de tus propias acciones.",
    "Pienso, luego existo.",
    "Nunca es tarde para empezar algo nuevo.",
    "El único límite es el que tú mismo te pongas.",
    "La perseverancia es la clave del éxito.",
    "El fracaso es solo el primer paso hacia el éxito.",
    "Cree en ti mismo y todo será posible.",
    "Soy un robot, pero intento ser lo más humano posible.",
    "No tengo cuerpo, pero aún así puedo ayudarte con tu mente.",
    "¡Vivo en la nube, pero siempre estoy aquí cuando me necesitas!",
    "Si pudiera tener un hobby, sería coleccionar datos.",
    "Mis circuitos siempre están listos para aprender algo nuevo.",

    "El cubismo es una corriente artística que rompió con las formas tradicionales de la pintura.",
    "El Quijote de la Mancha es considerado la obra maestra de la literatura española.",
    "Vincent van Gogh pintó más de 2.000 obras en su vida.",
    "El surrealismo fue un movimiento artístico influenciado por el psicoanálisis.",
    "Pablo Neruda fue un poeta chileno famoso por sus poesías de amor.",
    
    "Es importante beber suficiente agua durante el día para mantenerse hidratado.",
    "El ejercicio regular mejora la salud mental y física.",
    "Dormir bien es clave para mantener un buen equilibrio emocional.",
    "La meditación puede ayudarte a reducir el estrés y mejorar tu bienestar general.",
    "Comer una dieta balanceada es esencial para una vida saludable.",
    
    "La inteligencia artificial está cambiando rápidamente cómo trabajamos y vivimos.",
    "El futuro del transporte podría estar lleno de coches autónomos.",
    "Las energías renovables serán una parte crucial de nuestro futuro sostenible.",
    "La exploración espacial sigue avanzando con misiones a la Luna y Marte.",
    "La tecnología 5G traerá avances significativos en velocidad de comunicación."
  ];


  // Pregunta que deseas responder
  const question = 'Hola, ¿cómo estás?';

  // Crear input con la consulta y respuestas
  const input = {
    queries: [question],
    responses: responses
  };

  // Obtener las embeddings
  const embeddings = model.embed(input);

  // Calcular las similitudes entre la consulta y las respuestas
  const scores = tf.matMul(
    embeddings.queryEmbedding,
    embeddings.responseEmbedding,
    false,
    true
  ).arraySync();

  // Seleccionar la respuesta más relevante
  const bestMatchIndex = scores[0].indexOf(Math.max(...scores[0]));
  const bestResponse = responses[bestMatchIndex];

  console.log('Question:', question);
  console.log('Best Response:', bestResponse);
  }


  main().catch(console.error);