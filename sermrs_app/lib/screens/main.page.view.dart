import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:sermrs_app/models/music.dart';
import 'home.dart';
import 'library.dart';
import 'profile.dart';
import 'search.dart';
import 'settings.dart';

class MainPage extends StatefulWidget {
  const MainPage({Key? key}) : super(key: key);

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  final AudioPlayer _audioPlayer = AudioPlayer();
  var tabs = [];
  int currentTabIndex = 0;
  bool isPlaying = false;
  Music? music;

  Widget miniPlayer(Music? music, {bool stop = false}) {
    this.music = music;
    setState(() {});
    if (music == null) {
      return const SizedBox();
    }
    if (stop) {
      isPlaying = false;
      _audioPlayer.stop();
    }
    setState(() {});
    Size deviceSize = MediaQuery.of(context).size;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 500),
      color: Colors.blueGrey,
      width: deviceSize.width,
      height: 50,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Image.network(music.image, fit: BoxFit.cover),
          Text(
            music.name,
            style: const TextStyle(color: Colors.white),
          ),
          IconButton(
              onPressed: () async {
                  isPlaying = !isPlaying;
                  if (isPlaying) {
                    await _audioPlayer.play(UrlSource(music.audioURL));
                  } else {
                    await _audioPlayer.pause();
                  }
                  setState(() {});
                },
              icon: isPlaying
                  ? const Icon(Icons.pause, color: Colors.white)
                  : const Icon(Icons.play_arrow, color: Colors.white))
        ],
      ),
    );
  }

  @override
  void initState() {
    super.initState();
    tabs = [
      Home(miniPlayer),
      const Search(),
      const Library(),
      const Profile(),
      const Settings(),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: tabs[currentTabIndex],
      backgroundColor: Colors.black,
      bottomNavigationBar: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          miniPlayer(music),
          BottomNavigationBar(
            currentIndex: currentTabIndex,
            onTap: (currentIndex) {
              currentTabIndex = currentIndex;
              setState(() {}); //rerender
            },
            selectedLabelStyle: const TextStyle(color: Colors.white),
            selectedFontSize: 10,
            items: const [
              BottomNavigationBarItem(
                  backgroundColor: Colors.blueGrey,
                  icon: Icon(Icons.home, color: Colors.white),
                  label: 'Home'),
              BottomNavigationBarItem(
                  backgroundColor: Colors.blueGrey,
                  icon: Icon(Icons.search, color: Colors.white),
                  label: 'Search'),
              BottomNavigationBarItem(
                  backgroundColor: Colors.blueGrey,
                  icon: Icon(Icons.music_note, color: Colors.white),
                  label: 'Library'),
              BottomNavigationBarItem(
                  backgroundColor: Colors.blueGrey,
                  icon: Icon(Icons.person, color: Colors.white),
                  label: 'Profile'),
              BottomNavigationBarItem(
                  backgroundColor: Colors.blueGrey,
                  icon: Icon(Icons.settings, color: Colors.white),
                  label: 'Settings'),
            ],
          ),
        ],
      ),
    );
  }
}
