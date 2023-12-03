import 'package:avatar_glow/avatar_glow.dart';
import 'package:flutter/material.dart';
import 'package:flutter_hooks/flutter_hooks.dart';

import 'main.page.view.dart';

class VoiceDectectScreen extends HookWidget {
  const VoiceDectectScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SizedBox(
        height: MediaQuery.of(context).size.height,
        width: MediaQuery.of(context).size.width,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              alignment: Alignment.topRight,
              child: CircleAvatar(
                radius: 20,
                backgroundColor: Colors.white,
                child: IconButton(
                  color: Colors.black,
                  onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => const MainPage()));
                  },
                  icon: const Icon(Icons.home, color: Colors.black),
                ),
              ),
            ),
            AvatarGlow(
              endRadius: 200.0,
              animate: true,
              child: GestureDetector(
                onTap: () => ('Tapped'),
                child: Material(
                  shape: const CircleBorder(),
                  elevation: 8,
                  child: Container(
                    padding: const EdgeInsets.all(40),
                    height: 200,
                    width: 200,
                    decoration: const BoxDecoration(
                        shape: BoxShape.circle, color: Colors.tealAccent),
                    child: Image.asset(
                      'assets/images/google.png',
                      color: Colors.white,
                    ),
                  ),
                ),
              ),
            ),
            const Text(
              'Tap to MoodTunes',
              style: TextStyle(color: Colors.white, fontSize: 40),
            ),
            const SizedBox(
              height: 40,
            ),
          ],
        ),
      ),
    );
  }
}
