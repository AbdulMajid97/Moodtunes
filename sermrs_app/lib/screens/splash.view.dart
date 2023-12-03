import 'dart:async';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

import 'login.view.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  State<SplashScreen> createState() => _SplashViewState();
}

class _SplashViewState extends State<SplashScreen> {
  @override
  Widget build(BuildContext context) {
    Timer(const Duration(seconds: 15), () {
      Get.to(const LogInScreen());
    });

    return const Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Text(
          'MoodTunes',
          style: TextStyle(
            color: Colors.white,
            fontSize: 50,
            fontWeight: FontWeight.bold,
            fontStyle: FontStyle.italic,
          ),
        ),
      ),
    );
  }
}
